"""
Built-in social simulation engine (replaces camel-ai/OASIS).

Runs entirely on the configured OpenAI-compatible LLM provider (OpenRouter).
No external simulation framework or memory service is required.

Usage:
    python run_simulation.py --config /path/to/simulation_config.json
    python run_simulation.py --config /path/to/simulation_config.json --no-wait

Outputs (in the simulation directory):
    - {platform}/actions.jsonl          action/event log consumed by the backend monitor
    - {platform}_simulation.db          SQLite DB with post/comment/trace/user tables
    - env_status.json                   IPC environment status
    - ipc_commands/ ipc_responses/      file-based IPC with the Flask backend
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import re
import signal
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, ".."))
_project_root = os.path.abspath(os.path.join(_backend_dir, ".."))
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, _backend_dir)

from dotenv import load_dotenv

_env_file = os.path.join(_project_root, ".env")
if os.path.exists(_env_file):
    load_dotenv(_env_file)
else:
    _backend_env = os.path.join(_backend_dir, ".env")
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)

from action_logger import PlatformActionLogger, SimulationLogManager

IPC_COMMANDS_DIR = "ipc_commands"
IPC_RESPONSES_DIR = "ipc_responses"
ENV_STATUS_FILE = "env_status.json"

TWITTER_ACTIONS = ["CREATE_POST", "LIKE_POST", "REPOST", "FOLLOW", "DO_NOTHING", "QUOTE_POST"]
REDDIT_ACTIONS = [
    "CREATE_POST",
    "CREATE_COMMENT",
    "LIKE_POST",
    "DISLIKE_POST",
    "FOLLOW",
    "DO_NOTHING",
]

MAX_CONCURRENT_LLM_CALLS = int(os.environ.get("SIMULATION_MAX_CONCURRENCY", "30"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(asctime)s - %(message)s")
logger = logging.getLogger("mirofish.simulation")


class LLMClient:
    """Minimal OpenAI-compatible chat client (OpenRouter by default)."""

    def __init__(self):
        from openai import OpenAI

        api_key = os.environ.get("LLM_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing API key: set LLM_API_KEY in the .env file at the project root")
        self.model = os.environ.get("LLM_MODEL_NAME", "openai/gpt-4o-mini")
        self._client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout=float(os.environ.get("LLM_TIMEOUT_SECONDS", "120")),
        )

    def chat(self, prompt: str, temperature: float = 0.8) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    return None


class SimulationDB:
    """SQLite store for one platform's simulated social feed."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        if os.path.exists(db_path):
            os.remove(db_path)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._next_post_id = 1
        self._next_comment_id = 1
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE user (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    name TEXT,
                    description TEXT
                );
                CREATE TABLE post (
                    post_id INTEGER PRIMARY KEY,
                    agent_id INTEGER,
                    content TEXT,
                    created_at TEXT,
                    likes INTEGER DEFAULT 0,
                    dislikes INTEGER DEFAULT 0,
                    reposts INTEGER DEFAULT 0
                );
                CREATE TABLE comment (
                    comment_id INTEGER PRIMARY KEY,
                    post_id INTEGER,
                    agent_id INTEGER,
                    content TEXT,
                    created_at TEXT
                );
                CREATE TABLE trace (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT,
                    user_id INTEGER,
                    info TEXT,
                    created_at TEXT
                );
                """
            )

    def add_user(self, user_id: int, username: str, name: str, description: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO user (user_id, username, name, description) VALUES (?, ?, ?, ?)",
                (user_id, username, name, description),
            )

    def add_post(self, agent_id: int, content: str, created_at: str) -> int:
        post_id = self._next_post_id
        self._next_post_id += 1
        with self.conn:
            self.conn.execute(
                "INSERT INTO post (post_id, agent_id, content, created_at) VALUES (?, ?, ?, ?)",
                (post_id, agent_id, content, created_at),
            )
        return post_id

    def add_comment(self, post_id: int, agent_id: int, content: str, created_at: str) -> int:
        comment_id = self._next_comment_id
        self._next_comment_id += 1
        with self.conn:
            self.conn.execute(
                "INSERT INTO comment (comment_id, post_id, agent_id, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (comment_id, post_id, agent_id, content, created_at),
            )
        return comment_id

    def like_post(self, post_id: int, delta: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE post SET likes = likes + ? WHERE post_id = ? AND ? > 0",
                (max(delta, 0), post_id, delta),
            )
            self.conn.execute(
                "UPDATE post SET dislikes = dislikes + 1 WHERE post_id = ? AND ? < 0",
                (post_id, delta),
            )

    def repost(self, post_id: int) -> None:
        with self.conn:
            self.conn.execute("UPDATE post SET reposts = reposts + 1 WHERE post_id = ?", (post_id,))

    def add_trace(self, action: str, user_id: int, info: Dict[str, Any], created_at: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO trace (action, user_id, info, created_at) VALUES (?, ?, ?, ?)",
                (action, user_id, json.dumps(info, ensure_ascii=False), created_at),
            )

    def recent_posts(self, limit: int = 15) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT post_id, agent_id, content, created_at, likes, reposts FROM post "
            "ORDER BY post_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_post(self, post_id: int) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM post WHERE post_id = ?", (post_id,)
        ).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        self.conn.close()


class PlatformSimulator:
    """Runs the action loop for one platform (twitter or reddit)."""

    def __init__(
        self,
        platform: str,
        config: Dict[str, Any],
        simulation_dir: str,
        llm: LLMClient,
        log_manager: SimulationLogManager,
    ):
        self.platform = platform
        self.config = config
        self.simulation_dir = simulation_dir
        self.llm = llm
        self.actions = TWITTER_ACTIONS if platform == "twitter" else REDDIT_ACTIONS
        self.action_logger: PlatformActionLogger = (
            log_manager.get_twitter_logger() if platform == "twitter" else log_manager.get_reddit_logger()
        )
        self.db = SimulationDB(os.path.join(simulation_dir, f"{platform}_simulation.db"))
        self.follows: Dict[int, set] = {}
        self.profiles: List[Dict[str, Any]] = []
        self.agents_by_id: Dict[int, Dict[str, Any]] = {}
        self._base_time = datetime.now()

    def load_profiles(self) -> bool:
        if self.platform == "twitter":
            path = os.path.join(self.simulation_dir, "twitter_profiles.csv")
            if not os.path.exists(path):
                return False
            with open(path, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    profile = {
                        "user_id": int(float(row.get("user_id", 0) or 0)),
                        "name": row.get("name", ""),
                        "username": row.get("username", ""),
                        "user_char": row.get("user_char", ""),
                        "description": row.get("description", ""),
                    }
                    self.profiles.append(profile)
        else:
            path = os.path.join(self.simulation_dir, "reddit_profiles.json")
            if not os.path.exists(path):
                return False
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("agents") or data.get("profiles") or []
            for item in data:
                persona = item.get("persona") or item.get("bio") or ""
                profile = {
                    "user_id": int(item.get("user_id", 0) or 0),
                    "name": item.get("name", ""),
                    "username": item.get("username", ""),
                    "user_char": f"{item.get('bio', '')} {persona}".strip(),
                    "description": item.get("bio", ""),
                }
                self.profiles.append(profile)

        for profile in self.profiles:
            self.agents_by_id[profile["user_id"]] = profile
            self.db.add_user(
                profile["user_id"],
                profile["username"],
                profile["name"],
                profile["description"],
            )
        return bool(self.profiles)

    def _simulated_time(self, round_num: int) -> str:
        minutes_per_round = self.config.get("time_config", {}).get("minutes_per_round", 30)
        return (self._base_time + timedelta(minutes=round_num * minutes_per_round)).isoformat()

    def _active_agents_for_round(self, simulated_hour: int) -> List[Dict[str, Any]]:
        time_config = self.config.get("time_config", {})
        agent_configs = self.config.get("agent_configs", [])

        base_min = time_config.get("agents_per_hour_min", 5)
        base_max = time_config.get("agents_per_hour_max", 20)
        peak_hours = time_config.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22])
        off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])
        if simulated_hour in peak_hours:
            multiplier = time_config.get("peak_activity_multiplier", 1.5)
        elif simulated_hour in off_peak_hours:
            multiplier = time_config.get("off_peak_activity_multiplier", 0.3)
        else:
            multiplier = 1.0
        target_count = int(random.uniform(base_min, base_max) * multiplier)

        config_by_id = {c.get("agent_id"): c for c in agent_configs}
        candidates = []
        for profile in self.profiles:
            cfg = config_by_id.get(profile["user_id"], {})
            active_hours = cfg.get("active_hours", list(range(8, 23)))
            activity_level = cfg.get("activity_level", 0.5)
            if simulated_hour not in active_hours:
                continue
            if random.random() < activity_level:
                candidates.append(profile)

        selected = random.sample(candidates, min(target_count, len(candidates))) if candidates else []
        return selected

    def _feed_prompt(self) -> str:
        posts = self.db.recent_posts()
        if not posts:
            return "The feed is currently empty."
        lines = []
        for post in reversed(posts):
            author = self.agents_by_id.get(post["agent_id"], {}).get("username", str(post["agent_id"]))
            lines.append(
                f"[post_id={post['post_id']}] @{author}: {post['content']} "
                f"(likes={post['likes']}, reposts={post['reposts']})"
            )
        return "\n".join(lines[-15:])

    def _decide_action(self, profile: Dict[str, Any], round_num: int) -> Dict[str, Any]:
        prompt = f"""You are simulating a social media user on {'Twitter' if self.platform == 'twitter' else 'Reddit'}.

Your persona:
{profile['user_char'] or profile['description']}

Current feed (most recent posts):
{self._feed_prompt()}

Choose ONE action to take right now. Available actions:
{json.dumps(self.actions)}

Respond ONLY with a JSON object:
{{"action_type": "<one of the available actions>", "content": "<text you write, if any>", "post_id": <id of an existing post, when the action refers to one>, "target_agent_id": <agent id you follow, for FOLLOW>}}

Rules:
- CREATE_POST / QUOTE_POST / CREATE_COMMENT require "content".
- LIKE_POST / DISLIKE_POST / REPOST require "post_id" from the feed.
- DO_NOTHING requires nothing else.
- Stay in character; write content as your persona actually would."""
        try:
            raw = self.llm.chat(prompt)
        except Exception as error:
            logger.warning("LLM action decision failed for agent %s: %s", profile["user_id"], error)
            return {"action_type": "DO_NOTHING"}
        parsed = parse_json_object(raw)
        if not parsed or parsed.get("action_type") not in self.actions:
            return {"action_type": "DO_NOTHING"}
        return parsed

    def _apply_action(self, profile: Dict[str, Any], decision: Dict[str, Any], round_num: int) -> Dict[str, Any]:
        action_type = decision.get("action_type", "DO_NOTHING")
        created_at = self._simulated_time(round_num)
        agent_id = profile["user_id"]
        args: Dict[str, Any] = {}
        result = "ok"

        if action_type == "CREATE_POST":
            content = str(decision.get("content") or "").strip()
            if not content:
                action_type, result = "DO_NOTHING", "empty content"
            else:
                post_id = self.db.add_post(agent_id, content, created_at)
                args = {"content": content, "post_id": post_id}
        elif action_type == "QUOTE_POST":
            content = str(decision.get("content") or "").strip()
            quoted = self.db.get_post(int(decision.get("post_id") or 0))
            if not content or not quoted:
                action_type, result = "DO_NOTHING", "missing quote content or post"
            else:
                post_id = self.db.add_post(agent_id, content, created_at)
                args = {"content": content, "quoted_post_id": quoted["post_id"], "post_id": post_id}
        elif action_type == "CREATE_COMMENT":
            content = str(decision.get("content") or "").strip()
            parent = self.db.get_post(int(decision.get("post_id") or 0))
            if not content or not parent:
                action_type, result = "DO_NOTHING", "missing comment content or post"
            else:
                comment_id = self.db.add_comment(parent["post_id"], agent_id, content, created_at)
                args = {"content": content, "post_id": parent["post_id"], "comment_id": comment_id}
        elif action_type == "LIKE_POST":
            post = self.db.get_post(int(decision.get("post_id") or 0))
            if not post:
                action_type, result = "DO_NOTHING", "post not found"
            else:
                self.db.like_post(post["post_id"], 1)
                args = {"post_id": post["post_id"]}
        elif action_type == "DISLIKE_POST":
            post = self.db.get_post(int(decision.get("post_id") or 0))
            if not post:
                action_type, result = "DO_NOTHING", "post not found"
            else:
                self.db.like_post(post["post_id"], -1)
                args = {"post_id": post["post_id"]}
        elif action_type == "REPOST":
            post = self.db.get_post(int(decision.get("post_id") or 0))
            if not post:
                action_type, result = "DO_NOTHING", "post not found"
            else:
                self.db.repost(post["post_id"])
                args = {"post_id": post["post_id"]}
        elif action_type == "FOLLOW":
            target_id = int(decision.get("target_agent_id") or -1)
            if target_id not in self.agents_by_id or target_id == agent_id:
                action_type, result = "DO_NOTHING", "invalid follow target"
            else:
                self.follows.setdefault(agent_id, set()).add(target_id)
                args = {"target_agent_id": target_id}

        self.action_logger.log_action(
            round_num=round_num,
            agent_id=agent_id,
            agent_name=profile.get("name") or profile.get("username") or f"Agent_{agent_id}",
            action_type=action_type,
            action_args=args,
            result=result,
            success=True,
        )
        if action_type != "DO_NOTHING":
            self.db.add_trace(
                action_type.lower(),
                agent_id,
                {"action_type": action_type, **args},
                created_at,
            )
        return {"action_type": action_type, "args": args}

    def publish_initial_posts(self) -> int:
        initial_posts = self.config.get("event_config", {}).get("initial_posts", [])
        published = 0
        for post in initial_posts:
            agent_id = int(post.get("poster_agent_id", 0) or 0)
            content = str(post.get("content") or "").strip()
            profile = self.agents_by_id.get(agent_id)
            if not profile or not content:
                continue
            post_id = self.db.add_post(agent_id, content, self._simulated_time(0))
            self.action_logger.log_action(
                round_num=0,
                agent_id=agent_id,
                agent_name=profile.get("name") or "",
                action_type="CREATE_POST",
                action_args={"content": content, "post_id": post_id},
                result="initial post",
                success=True,
            )
            published += 1
        return published

    async def run(self, total_rounds: int) -> int:
        self.action_logger.log_simulation_start(self.config)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
        total_actions = 0

        for round_num in range(total_rounds):
            time_config = self.config.get("time_config", {})
            minutes_per_round = time_config.get("minutes_per_round", 30)
            simulated_minutes = round_num * minutes_per_round
            simulated_hour = (simulated_minutes // 60) % 24

            self.action_logger.log_round_start(round_num, simulated_hour)
            active = self._active_agents_for_round(simulated_hour)

            async def run_agent(profile: Dict[str, Any]) -> Dict[str, Any]:
                async with semaphore:
                    return await asyncio.to_thread(self._decide_action, profile, round_num)

            decisions = await asyncio.gather(*(run_agent(p) for p in active))
            round_actions = 0
            for profile, decision in zip(active, decisions):
                outcome = self._apply_action(profile, decision, round_num)
                if outcome["action_type"] != "DO_NOTHING":
                    round_actions += 1
            total_actions += round_actions

            self.action_logger.log_round_end(round_num, round_actions)
            if (round_num + 1) % 10 == 0 or round_num == 0:
                logger.info(
                    "[%s] Round %s/%s - %s agents active - %s actions",
                    self.platform,
                    round_num + 1,
                    total_rounds,
                    len(active),
                    round_actions,
                )

        self.action_logger.log_simulation_end(total_rounds, total_actions)
        return total_actions

    def interview(self, agent_id: int, prompt: str) -> Dict[str, Any]:
        profile = self.agents_by_id.get(agent_id)
        if profile is None:
            return {"agent_id": agent_id, "response": None, "timestamp": None}
        full_prompt = (
            f"You are {profile.get('name') or profile.get('username')}, a simulated user.\n"
            f"Your persona: {profile['user_char'] or profile['description']}\n\n"
            f"Answer the following interview question in character:\n{prompt}"
        )
        try:
            answer = self.llm.chat(full_prompt, temperature=0.7)
        except Exception as error:
            answer = f"(interview failed: {error})"
        created_at = datetime.now().isoformat()
        info = {"prompt": prompt, "response": answer}
        self.db.add_trace("interview", agent_id, info, created_at)
        return {"agent_id": agent_id, "response": answer, "timestamp": created_at}


class IPCHandler:
    """File-based IPC compatible with the previous OASIS runner scripts."""

    def __init__(self, simulation_dir: str, simulators: Dict[str, PlatformSimulator]):
        self.simulation_dir = simulation_dir
        self.simulators = simulators
        self.commands_dir = os.path.join(simulation_dir, IPC_COMMANDS_DIR)
        self.responses_dir = os.path.join(simulation_dir, IPC_RESPONSES_DIR)
        self.status_file = os.path.join(simulation_dir, ENV_STATUS_FILE)
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)

    def update_status(self, status: str) -> None:
        with open(self.status_file, "w", encoding="utf-8") as f:
            json.dump({"status": status, "timestamp": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)

    def poll_command(self) -> Optional[Dict[str, Any]]:
        if not os.path.isdir(self.commands_dir):
            return None
        command_files = []
        for filename in os.listdir(self.commands_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.commands_dir, filename)
                command_files.append((filepath, os.path.getmtime(filepath)))
        command_files.sort(key=lambda item: item[1])
        for filepath, _ in command_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
        return None

    def send_response(self, command_id: str, status: str, result=None, error=None) -> None:
        response = {
            "command_id": command_id,
            "status": status,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(self.responses_dir, f"{command_id}.json"), "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        try:
            os.remove(os.path.join(self.commands_dir, f"{command_id}.json"))
        except OSError:
            pass

    def process_commands(self, shutdown_event: asyncio.Event) -> bool:
        command = self.poll_command()
        if not command:
            return True

        command_id = command.get("command_id")
        command_type = command.get("command_type")
        args = command.get("args", {})

        if command_type == "interview":
            platform = args.get("platform")
            simulator = self.simulators.get(platform) or next(iter(self.simulators.values()))
            result = simulator.interview(int(args.get("agent_id", 0)), args.get("prompt", ""))
            self.send_response(command_id, "completed", result=result)
            return True

        if command_type == "batch_interview":
            interviews = args.get("interviews", [])
            results = {}
            for interview in interviews:
                platform = interview.get("platform")
                simulator = self.simulators.get(platform) or next(iter(self.simulators.values()))
                agent_id = int(interview.get("agent_id", 0))
                results[agent_id] = simulator.interview(agent_id, interview.get("prompt", ""))
            self.send_response(command_id, "completed", result={"interviews_count": len(results), "results": results})
            return True

        if command_type == "close_env":
            self.send_response(command_id, "completed", result={"message": "The environment is about to close"})
            shutdown_event.set()
            return False

        self.send_response(command_id, "failed", error=f"Unknown command type: {command_type}")
        return True


class SimulationRunner:
    def __init__(self, config_path: str, wait_for_commands: bool = True):
        self.config_path = config_path
        self.config = self._load_config()
        self.simulation_dir = os.path.dirname(config_path)
        self.wait_for_commands = wait_for_commands

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _detect_platforms(self) -> List[str]:
        platforms = []
        if os.path.exists(os.path.join(self.simulation_dir, "twitter_profiles.csv")):
            platforms.append("twitter")
        if os.path.exists(os.path.join(self.simulation_dir, "reddit_profiles.json")):
            platforms.append("reddit")
        return platforms

    async def run(self, max_rounds: Optional[int] = None) -> None:
        print("=" * 60)
        print("Built-in multi-agent social simulation (OpenRouter LLM)")
        print(f"Config file: {self.config_path}")
        print(f"Simulation ID: {self.config.get('simulation_id', 'unknown')}")
        print("=" * 60)

        platforms = self._detect_platforms()
        if not platforms:
            print("Error: twitter_profiles.csv or reddit_profiles.json not found")
            return

        time_config = self.config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        total_rounds = (total_hours * 60) // minutes_per_round
        if max_rounds is not None and max_rounds > 0:
            total_rounds = min(total_rounds, max_rounds)

        print(f"Platforms: {platforms}")
        print(f"Total rounds: {total_rounds}")

        llm = LLMClient()
        log_manager = SimulationLogManager(self.simulation_dir)
        simulators: Dict[str, PlatformSimulator] = {}
        for platform in platforms:
            simulator = PlatformSimulator(platform, self.config, self.simulation_dir, llm, log_manager)
            if not simulator.load_profiles():
                print(f"Warning: failed to load {platform} profiles, skipping platform")
                continue
            simulators[platform] = simulator

        if not simulators:
            print("Error: no usable simulation platform")
            return

        ipc = IPCHandler(self.simulation_dir, simulators)
        ipc.update_status("running")

        for platform, simulator in simulators.items():
            published = simulator.publish_initial_posts()
            if published:
                print(f"  [{platform}] published {published} initial posts")

        print("\nStarting simulation loop...")
        start_time = datetime.now()
        await asyncio.gather(*(simulator.run(total_rounds) for simulator in simulators.values()))
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\nSimulation loop complete! Elapsed: {elapsed:.1f}s")

        if self.wait_for_commands:
            print("\nEntering wait-for-commands mode - environment stays alive")
            shutdown_event = asyncio.Event()
            ipc.update_status("alive")
            try:
                while not shutdown_event.is_set():
                    if not ipc.process_commands(shutdown_event):
                        break
                    try:
                        await asyncio.wait_for(shutdown_event.wait(), timeout=0.5)
                    except asyncio.TimeoutError:
                        pass
            except KeyboardInterrupt:
                print("\nInterrupt received")

        ipc.update_status("stopped")
        for simulator in simulators.values():
            simulator.db.close()
        print("Environment closed")
        print("=" * 60)


_shutdown_event: Optional[asyncio.Event] = None


def setup_signal_handlers():
    def signal_handler(signum, frame):
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\nReceived {sig_name}, exiting...")
        if _shutdown_event is not None:
            _shutdown_event.set()
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


async def main():
    global _shutdown_event
    parser = argparse.ArgumentParser(description="Built-in multi-agent social simulation (OpenRouter LLM)")
    parser.add_argument("--config", type=str, required=True, help="Config file path (simulation_config.json)")
    parser.add_argument("--max-rounds", type=int, default=None, help="Maximum simulation rounds (optional)")
    parser.add_argument("--no-wait", action="store_true", default=False, help="Close the environment immediately after simulation")
    args = parser.parse_args()

    _shutdown_event = asyncio.Event()
    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    runner = SimulationRunner(config_path=args.config, wait_for_commands=not args.no_wait)
    await runner.run(max_rounds=args.max_rounds)


if __name__ == "__main__":
    setup_signal_handlers()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("Simulation process exited")
