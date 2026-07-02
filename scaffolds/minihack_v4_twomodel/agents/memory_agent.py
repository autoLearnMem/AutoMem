import copy
import logging
import os
import re
from collections import deque

from omegaconf import OmegaConf

from agents.base import BaseAgent
from client import LLMResponse
from prompt_builder.history import Message

logger = logging.getLogger(__name__)

MAX_LOG_ITERATIONS = 5
MAX_VALIDATE_ITERATIONS = 5
MAX_SEARCH_RESULTS = 100          

FALLBACK_ACTIONS = {
    "nle": "search", "minihack": "search", "crafter": "noop",
}
FALLBACK_ACTION_DEFAULT = "wait"

_DIRECTION_COMMANDS = frozenset([
    "open", "close", "kick", "fight", "fire", "zap",
])
_ITEM_COMMANDS = frozenset([
    "eat", "quaff", "wear", "wield", "read", "apply", "puton", "remove",
    "takeoff", "dip", "rub", "invoke", "quiver", "throw", "offer", "drop",
    "force", "loot", "cast", "pickup",
])
_TWO_STEP_COMMANDS = _DIRECTION_COMMANDS | _ITEM_COMMANDS
_ALL_DIRECTIONS = frozenset([
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
])

_OPPOSITE_DIRECTIONS = {
    "north": "south", "south": "north",
    "east": "west", "west": "east",
    "northeast": "southwest", "southwest": "northeast",
    "northwest": "southeast", "southeast": "northwest",
}

_ADJACENT_DIRECTIONS = {
    "north": ("northeast", "northwest"),
    "south": ("southeast", "southwest"),
    "east": ("northeast", "southeast"),
    "west": ("northwest", "southwest"),
    "northeast": ("north", "east"),
    "northwest": ("north", "west"),
    "southeast": ("south", "east"),
    "southwest": ("south", "west"),
}

_PERPENDICULAR_DIRECTIONS = {
    "north": ("east", "west"),
    "south": ("east", "west"),
    "east": ("north", "south"),
    "west": ("north", "south"),
    "northeast": ("northwest", "southeast"),
    "northwest": ("northeast", "southwest"),
    "southeast": ("northeast", "southwest"),
    "southwest": ("northwest", "southeast"),
}

def repair_combined_action(raw_action):
    stripped = raw_action.strip()
    parts = stripped.split(None, 1)
    if len(parts) < 2:
        return stripped, False
    cmd = parts[0].lower()
    rest = parts[1].strip().lower()
    
    if cmd == "far" and rest in _ALL_DIRECTIONS:
        return stripped, False
    
    if cmd in ("go", "move") and rest in _ALL_DIRECTIONS:
        return rest, True
    if cmd in _TWO_STEP_COMMANDS:
        return cmd, True
    return stripped, False

class ActionHistoryTracker:
    def __init__(self, max_history=12):
        self.actions = deque(maxlen=max_history)

    def record(self, action):
        self.actions.append(action)

    def reset(self):
        self.actions.clear()

    def get_recent_actions_text(self):
        if not self.actions:
            return ""
        lines = []
        lines.append("RECENT ACTION HISTORY (last {} actions):".format(len(self.actions)))
        for i, act in enumerate(self.actions):
            lines.append("  {}: {}".format(i + 1, act))
        return "\n".join(lines)

    def detect_oscillation(self):
        if len(self.actions) < 4:
            return None
        recent = list(self.actions)[-6:]
        osc_count = 0
        for i in range(len(recent) - 1):
            if _OPPOSITE_DIRECTIONS.get(recent[i]) == recent[i + 1]:
                osc_count += 1
        if osc_count >= 2:
            last = recent[-1]
            opp = _OPPOSITE_DIRECTIONS.get(last, "")
            if opp:
                return (last, opp)
        return None

    def detect_repetition(self):
        if len(self.actions) < 3:
            return None
        recent = list(self.actions)[-3:]
        if recent[0] == recent[1] == recent[2]:
            return recent[0]
        return None

    def get_warnings(self, valid_directions=None):
        warnings = []
        osc = self.detect_oscillation()
        if osc:
            dir_a, dir_b = osc
            
            avoid = {dir_a, dir_b}
            if valid_directions is not None:
                alternatives = [d for d in valid_directions if d not in avoid]
            else:
                alternatives = [d for d in ["north", "south", "east", "west",
                                            "northeast", "northwest", "southeast", "southwest"]
                               if d not in avoid]
            alt_text = ", ".join(alternatives[:4])
            warnings.append(
                "OSCILLATION DETECTED: You have been alternating between '{}' and '{}'. This achieves NOTHING and wastes steps. You MUST choose a COMPLETELY DIFFERENT direction. Try one of: {}".format(dir_a, dir_b, alt_text)
            )
        rep = self.detect_repetition()
        if rep:
            warnings.append(
                "REPETITION DETECTED: You have done '{}' three times in a row with no progress. Try a different action.".format(rep)
            )
        return warnings

class PositionTracker:
    _POS_OFFSETS = {
        "north": (0, -1), "south": (0, 1),
        "east": (1, 0), "west": (-1, 0),
        "northeast": (1, -1), "northwest": (-1, -1),
        "southeast": (1, 1), "southwest": (-1, 1),
    }

    def __init__(self):
        self.positions = []
        self.visit_counts = {}
        self.current_pos = None
        
        self._consecutive_unchanged = 0
        self._last_mandatory_dir = None
        self._failed_mandatory_dirs = set()

    def reset(self):
        self.positions.clear()
        self.visit_counts.clear()
        self.current_pos = None
        self._consecutive_unchanged = 0
        self._last_mandatory_dir = None
        self._failed_mandatory_dirs.clear()

    def update(self, obs_text):
        m = re.search(r'\(x=(\d+),\s*y=(\d+)\)', obs_text)
        if m:
            pos = (int(m.group(1)), int(m.group(2)))
            old_pos = self.current_pos
            self.current_pos = pos
            self.positions.append(pos)
            self.visit_counts[pos] = self.visit_counts.get(pos, 0) + 1

            if old_pos is not None and pos == old_pos:
                self._consecutive_unchanged += 1

                if self._last_mandatory_dir is not None:
                    self._failed_mandatory_dirs.add(self._last_mandatory_dir)
            else:
                self._consecutive_unchanged = 0
                
                self._failed_mandatory_dirs.clear()

    def get_mandatory_direction(self, passable_directions):
        if not self.current_pos:
            self._last_mandatory_dir = None
            return None
        visit_count = self.visit_counts.get(self.current_pos, 1)
        if visit_count < 4:
            self._last_mandatory_dir = None
            return None

        x, y = self.current_pos

        effective_passable = [d for d in passable_directions
                             if d not in self._failed_mandatory_dirs]

        if not effective_passable:
            
            self._failed_mandatory_dirs.clear()
            effective_passable = list(passable_directions)

        unvisited = []
        for d in effective_passable:
            offset = self._POS_OFFSETS.get(d)
            if offset:
                target = (x + offset[0], y + offset[1])
                if self.visit_counts.get(target, 0) == 0:
                    unvisited.append(d)

        if unvisited:
            self._last_mandatory_dir = unvisited[0]
            return unvisited[0]

        if visit_count >= 5:
            min_visits = float('inf')
            min_dir = None
            for d in effective_passable:
                offset = self._POS_OFFSETS.get(d)
                if offset:
                    target = (x + offset[0], y + offset[1])
                    v = self.visit_counts.get(target, 0)
                    if 0 < v < min_visits:
                        min_visits = v
                        min_dir = d
            if min_dir:
                self._last_mandatory_dir = min_dir
                return min_dir

        self._last_mandatory_dir = None
        return None

    def get_exploration_guidance(self, passable_directions):
        if not self.current_pos:
            return ""

        x, y = self.current_pos
        visit_count = self.visit_counts.get(self.current_pos, 1)
        total_unique = len(self.visit_counts)
        total_steps = len(self.positions)

        lines = []
        lines.append("EXPLORATION STATUS:")
        lines.append("  Current position: ({}, {}), visited {} time(s)".format(
            x, y, visit_count))
        lines.append("  Explored: {} unique tiles in {} steps".format(
            total_unique, total_steps))

        unvisited_passable = []
        visited_passable = []
        for d in passable_directions:
            offset = self._POS_OFFSETS.get(d)
            if offset:
                target = (x + offset[0], y + offset[1])
                target_visits = self.visit_counts.get(target, 0)
                if target_visits == 0:
                    unvisited_passable.append(d)
                else:
                    visited_passable.append("{} ({}x)".format(d, target_visits))

        if unvisited_passable:
            lines.append("  UNVISITED passable directions: {}".format(
                ", ".join(unvisited_passable)))
            lines.append("  >>> PRIORITY: Move {} to explore NEW territory!".format(
                " or ".join(unvisited_passable[:3])))
        else:
            lines.append("  All adjacent passable tiles already visited.")
            
            min_visits = float('inf')
            min_dir = None
            for d in passable_directions:
                offset = self._POS_OFFSETS.get(d)
                if offset:
                    target = (x + offset[0], y + offset[1])
                    v = self.visit_counts.get(target, 0)
                    if 0 < v < min_visits:
                        min_visits = v
                        min_dir = d
            if min_dir:
                lines.append("  Least-visited direction: {} ({}x). Try going there.".format(
                    min_dir, min_visits))
            
            lines.append("  TIP: All local tiles explored. Look for DOORS ON MAP or")
            lines.append("  distant corridors to reach new areas.")

        if visited_passable:
            lines.append("  Already-visited: {}".format(", ".join(visited_passable)))

        if visit_count >= 3:
            lines.append("")
            lines.append("  *** You have been at ({},{}) {} times! Move to NEW tiles!".format(
                x, y, visit_count))

        if len(self.positions) >= 6:
            recent_6 = self.positions[-6:]
            unique_6 = len(set(recent_6))
            if unique_6 <= 2:
                lines.append("")
                lines.append(
                    "  *** CRITICAL OSCILLATION: Only {} unique positions in last 6 steps!".format(unique_6))
                pos_strs = ["({},{})".format(p[0], p[1]) for p in recent_6]
                lines.append("  Path: {}".format(" -> ".join(pos_strs)))
                lines.append(
                    "  You MUST break this pattern. Choose a DIFFERENT direction!")
                if unvisited_passable:
                    lines.append("  Suggested: {}".format(unvisited_passable[0]))
            elif unique_6 <= 3 and len(self.positions) >= 10:
                recent_10 = self.positions[-10:]
                unique_10 = len(set(recent_10))
                if unique_10 <= 3:
                    lines.append("")
                    lines.append(
                        "  *** STUCK: Only {} unique positions in last 10 steps.".format(unique_10))
                    lines.append(
                        "  Try a direction you have NOT used recently.")

        return "\n".join(lines)

class StairsNavigator:
    def __init__(self):
        self.stairs_direction = None
        self.stairs_distance = None
        self.stairs_adjacent_direction = None

    def reset(self):
        self.stairs_direction = None
        self.stairs_distance = None
        self.stairs_adjacent_direction = None

    def update_from_map_analysis(self, map_analysis_text):
        self.stairs_adjacent_direction = None
        self.stairs_direction = None
        self.stairs_distance = None

        m = re.search(r"(\w+)\s*->\s*'>'\s*STAIRS DOWN", map_analysis_text)
        if m:
            self.stairs_adjacent_direction = m.group(1).lower()
            self.stairs_direction = self.stairs_adjacent_direction
            self.stairs_distance = 1
            return

        m = re.search(r"'>'\s*seen\s*~(\d+)\s*tiles\s+(\w+)", map_analysis_text)
        if m:
            self.stairs_distance = int(m.group(1))
            self.stairs_direction = m.group(2).lower()

    def get_stairs_directive(self, passable_directions=None, map_analysis_text=""):
        if not self.stairs_direction:
            return ""

        direction = self.stairs_direction
        passable_set = set(passable_directions) if passable_directions else set()

        if self.stairs_adjacent_direction:
            return (
                "***************************************************\n"
                "*** STAIRS DOWN '>' IS DIRECTLY {}! ***\n"
                "*** Move {} RIGHT NOW to complete the level! ***\n"
                "***************************************************\n"
                "Output: <|ACTION|>{}<|END|>"
            ).format(
                direction.upper(), direction, direction
            )

        danger_pattern = re.findall(
            r"(\w+)\s*->\s*'\}'\s*DANGEROUS", map_analysis_text)
        danger_dirs = set(d.lower() for d in danger_pattern)

        goal_and_adj = {direction}
        adj1, adj2 = _ADJACENT_DIRECTIONS.get(direction, ("", ""))
        if adj1:
            goal_and_adj.add(adj1)
        if adj2:
            goal_and_adj.add(adj2)
        lava_blocked_count = len(goal_and_adj & danger_dirs)

        if lava_blocked_count >= 2:
            
            perp_dirs = _PERPENDICULAR_DIRECTIONS.get(direction, ())
            perp_passable = [d for d in perp_dirs if d in passable_set]
            if not perp_passable:
                perp_passable = list(perp_dirs)

            return (
                "*** LAVA BARRIER BLOCKS PATH TO STAIRS ***\n"
                "Stairs '>' is ~{dist} tiles {dir}, but LAVA blocks the way.\n"
                "You CANNOT walk through lava -- it is DEADLY.\n"
                "DETOUR REQUIRED: Go {perp} to find a way AROUND the lava.\n"
                "Keep going {perp} until you find an opening, then head {dir}.\n"
                "Do NOT oscillate near the lava. Commit to the detour!"
            ).format(
                dist=self.stairs_distance,
                dir=direction,
                perp=" or ".join(perp_passable[:2]),
            )

        alt1, alt2 = _ADJACENT_DIRECTIONS.get(direction, ("", ""))
        
        passable_alts = [d for d in (direction, alt1, alt2) if d in passable_set]
        if not passable_alts and passable_set:
            
            passable_alts = list(passable_set)[:3]

        alt_text = ""
        if passable_alts:
            alt_text = "Passable directions toward goal: {}".format(
                ", ".join(passable_alts))

        return (
            "*** GOAL: Stairs down '>' is ~{dist} tiles {dir}. ***\n"
            "Navigate toward {dir}. Every step should bring you closer.\n"
            "{alts}\n"
            "If {dir} is blocked, try {alt1} or {alt2} to get around obstacles."
        ).format(
            dist=self.stairs_distance,
            dir=direction,
            alts=alt_text,
            alt1=alt1,
            alt2=alt2,
        )

def _extract_passable_directions(map_analysis_text):
    m = re.search(r'CAN MOVE:\s*(.*)', map_analysis_text)
    if not m:
        return []
    text = m.group(1)
    dirs = []
    
    for d in ["northeast", "northwest", "southeast", "southwest",
              "north", "south", "east", "west"]:
        if re.search(r'\b' + d + r'\b', text):
            dirs.append(d)
    return dirs

def _extract_blocked_directions(map_analysis_text):
    m = re.search(r'BLOCKED:\s*(.*)', map_analysis_text)
    if not m:
        return []
    text = m.group(1)
    dirs = []
    for d in ["northeast", "northwest", "southeast", "southwest",
              "north", "south", "east", "west"]:
        if re.search(r'\b' + d + r'\b', text):
            dirs.append(d)
    return dirs

def _detect_game_prompt(obs_text):
    if "In what direction?" in obs_text:
        return (
            "*** GAME PROMPT: The game asked 'In what direction?' ***\n"
            "Your previous command (e.g. 'open', 'kick', 'fight') needs a "
            "DIRECTION to complete.\n"
            "You MUST output ONLY a direction word: north, south, east, west, "
            "northeast, northwest, southeast, southwest.\n"
            "Do NOT output another command like 'open' or 'search'."
        )
    
    for prompt_text in ["What do you want to eat?",
                        "What do you want to quaff?",
                        "What do you want to wield?",
                        "What do you want to wear?",
                        "What do you want to read?"]:
        if prompt_text in obs_text:
            return (
                "*** GAME PROMPT: The game asked '{}' ***\n"
                "Respond with ONLY the single letter of the item "
                "(e.g., 'a', 'b', 'c').\n"
                "Do NOT output a full command."
            ).format(prompt_text)
    return ""

def _is_game_prompt_active(obs_text):
    return ("In what direction?" in obs_text or
            "What do you want to eat?" in obs_text or
            "What do you want to quaff?" in obs_text or
            "What do you want to wield?" in obs_text or
            "What do you want to wear?" in obs_text or
            "What do you want to read?" in obs_text)

def get_task_tips(task_name):
    if not task_name:
        return ""
    task_lower = task_name.lower()
    if "boxoban" in task_lower:
        return (
            "TASK-SPECIFIC RULES (Boxoban/Sokoban puzzle):\n"
            "- ONLY cardinal directions are valid: north, south, east, west.\n"
            "- Diagonal moves (northeast, etc.) are INVALID and will be rejected.\n"
            "- 'search', 'open', 'kick' are also INVALID in this task.\n"
            "- '#' symbols on the map are IRON BARS -- you CANNOT walk through them.\n"
            "- Push boulders ('`') by walking into them. Plan pushes carefully.\n"
            "- A boulder against a wall cannot be pushed further that way.\n"
            "- If a direction doesn't work, try a DIFFERENT one immediately."
        )
    elif "mazewalk" in task_lower:
        return (
            "TASK-SPECIFIC TIPS (Maze exploration):\n"
            "- Your goal is to find and reach the '>' stairs down.\n"
            "- Explore systematically using DFS: go forward until blocked, "
            "then backtrack and try a new path.\n"
            "- If you see '>' on the map or in the observation, move toward "
            "it IMMEDIATELY -- this is your #1 priority.\n"
            "- NEVER oscillate back and forth. If you've been somewhere, "
            "try a DIFFERENT direction.\n"
            "- When EXPLORATION STATUS shows UNVISITED directions, you MUST "
            "go to one of them. This is not optional.\n"
            "- If your action has NO EFFECT (observation unchanged), that direction "
            "is BLOCKED -- try a different one immediately."
        )
    elif "corridorbattle" in task_lower:
        return (
            "TASK-SPECIFIC TIPS (Corridor battle):\n"
            "- Navigate dark corridors and fight monsters to reach '>' stairs.\n"
            "- To attack a visible monster, move INTO its tile.\n"
            "- FIGHT-AND-ADVANCE: When fighting, keep moving TOWARD the goal.\n"
            "  Moving into a monster's tile attacks it AND advances your position.\n"
            "- Do NOT stand still while fighting -- keep advancing east/northeast.\n"
            "- Do NOT avoid monsters -- you must FIGHT them or they will\n"
            "  surround and kill you.\n"
            "- In DARK areas, the language observation describes entities beyond\n"
            "  your visible range -- trust it!\n"
            "- If HP is low, try eating food ('eat' then item letter).\n"
            "- Look for '>' stairs down to complete the level."
        )
    elif "corridor" in task_lower:
        return (
            "TASK-SPECIFIC TIPS (Corridor navigation):\n"
            "- Navigate through connected rooms via corridors and doors.\n"
            "- Doors '+' must be opened in TWO separate steps:\n"
            "  Step 1: Output 'open'\n"
            "  Step 2: When game asks 'In what direction?', output ONLY the "
            "direction (e.g. 'east'). Do NOT output 'open' again.\n"
            "- If a door is locked, use 'kick' then direction.\n"
            "- If stuck, try 'search' to find hidden doors or passages.\n"
            "- '>' stairs DOWN is your goal. '<' stairs UP is NOT the goal.\n"
            "- DOORS are your KEY to progress! When you see '+' on the map,\n"
            "  navigate toward it and open it. Check DOORS ON MAP in map analysis.\n"
            "- After opening a door, walk THROUGH it to the other side.\n"
            "- Move systematically: explore unvisited directions first."
        )
    elif "quest" in task_lower:
        return (
            "TASK-SPECIFIC TIPS (Quest):\n"
            "- Find and reach the '>' stairs down -- this is your primary goal.\n"
            "- If stairs are visible on the map, navigate TOWARD them persistently.\n"
            "- To attack monsters, move into their tile. Fight immediately.\n"
            "- LAVA ('}' symbol) is DEADLY and IMPASSABLE. Do NOT walk into lava.\n"
            "- If lava blocks your path, you MUST detour AROUND it:\n"
            "  Go far north or south along the lava until you find an opening,\n"
            "  then head back toward the stairs. Commit to the detour!\n"
            "- Do NOT oscillate near the lava barrier. Pick a direction and go.\n"
            "- Explore systematically: always prefer UNVISITED directions.\n"
            "- Do NOT circle near the start. Push outward toward the stairs.\n"
            "- Watch HP! If low, try eating food or praying."
        )
    return ""

NLE_GAME_RULES = """\
ACTION FORMAT RULES -- Read this before acting!

MOVEMENT: Output a single direction word: north, south, east, west, northeast, northwest, southeast, southwest. These move exactly 1 tile. You can also output 'far north', 'far east', far south', 'far west', 'far northeast', 'far southeast', 'far southwest', 'far northwest' to sprint in that direction until hitting an obstacle (wall, monster, item, door).

MELEE COMBAT: To attack an adjacent monster, simply MOVE INTO its tile. Example: if a goblin is to your east, output 'east' to attack it. This is the primary way to fight. Do NOT output 'fight east'.

FIGHT COMMAND: 'fight' is ONLY for attacking a monster you believe is there but CANNOT see (invisible/hidden). After outputting 'fight', the game will ask "In what direction?" -- respond with JUST the direction.

RANGED COMBAT:
- fire: shoots your quivered ammunition. Two-step: output 'fire', then the direction when prompted. Uses whatever is in your quiver automatically.
- quiver: select ammunition for your quiver. Two-step: output 'quiver', then the item letter when prompted. Do this before using 'fire'.
- throw: throw any item. Three-step: output 'throw', then the item letter, then the direction. Works for daggers, darts, potions, etc.
- zap: use a wand. Multi-step: output 'zap', then the wand letter, then the direction (for directional wands like fire/cold/death). Some wands do not ask for direction.

DOORS: Output 'open'. The game then asks "In what direction?" -- respond with the direction word. For locked doors, try 'kick' then direction.

STAIRS: You must be standing ON the stairs symbol. Then output 'up' for '<' stairs, or 'down' for '>' stairs. IMPORTANT: At Dungeon Level 1 (Dlvl:1), the '<' staircase leads OUT of the dungeon and ENDS THE GAME. Your goal is to go DEEPER -- find '>' stairs down. Only go 'up' at Dlvl:1 if you want to quit.

EATING: Output 'eat'. The game asks "What do you want to eat? [abc or ?*]". Respond with JUST the single letter of the food item. Do NOT combine them -- 'eat d' is INVALID; first 'eat', then 'd'.

ALL ITEM COMMANDS: eat, quaff, wear, wield, read, apply, puton, remove, takeoff, dip, rub, invoke -- these are ALL two-step: first output the command word alone, then respond with the item letter when the game prompts you. NEVER combine command + letter in one action.

PICKING UP ITEMS: Items on the floor are collected AUTOMATICALLY when you walk onto them -- you will see a message assigning an inventory letter. Gold ($) is always collected instantly. You do NOT need to output 'pickup' in normal play. Use 'pickup' only if multiple items are on one tile and you need to choose, or to retrieve something you previously dropped. Use 'loot' to take items from containers (boxes, chests) on the floor.

SEARCHING: Output 'search' to look for hidden doors/passages. You may need to search the same spot multiple times.

DIRECTION PROMPTS: When the game asks "In what direction?", respond with a direction word (north, south, east, west, northeast, etc.).

ITEM PROMPTS: When the game asks "What do you want to X? [abc or ?*]", respond with JUST the single letter of the item.

PRAYING: In desperate situations (very low HP, starving), 'pray' can save you. But do not pray more than once every ~300 turns or your god may get angry.

ENGRAVING: Writing 'Elbereth' on the ground can scare away many monsters. Use 'engrave' then follow the prompts.
"""

NLE_SYMBOLS = """\
MAP SYMBOLS -- Reference for reading the ASCII map.

TERRAIN:
  @ = you (the player)
  . = floor / open ground (walkable)
  # = corridor (walkable) -- BUT in Boxoban/Sokoban, # = iron bars (BLOCKED)
  (space) = rock / solid stone (CANNOT walk through)
  - = horizontal wall (CANNOT walk through)
  | = vertical wall (CANNOT walk through)
  + = closed door (use 'open' + direction) OR spellbook on floor
  < = staircase up (use 'up' when standing on it)
  > = staircase down (use 'down' when standing on it)
  ^ = trap (DANGEROUS -- try to avoid stepping on it)
  } = water / moat / lava (dangerous, may drown or burn)
  { = fountain (can 'quaff' from it)
  _ = altar (can 'offer' sacrifices)
  \\ = throne
  ` = boulder

ITEMS ON FLOOR:
  $ = gold (auto-collected by walking on it)
  ) = weapon
  [ = armor / shield
  ! = potion
  ? = scroll
  / = wand
  = = ring
  " = amulet
  ( = tool
  * = gem or rock
  % = food or corpse (can eat)
  ~ = iron chain or whip

COMMON MONSTER LETTERS (lowercase = small, UPPERCASE = large):
  a = ant / insect
  b = blob
  d = canine (dog, jackal, wolf)
  e = floating eye -- do NOT melee attack (paralyzes you!)
  f = feline (cat, panther, tiger)
  g = gremlin / gnome (lowercase)
  h = humanoid (dwarf, mind flayer)
  i = imp / minor demon
  k = kobold
  o = orc
  r = rodent (rat, rock mole)
  s = spider / scorpion
  u = horse / unicorn
  w = worm
  D = dragon (very dangerous)
  G = gnome (capital)
  H = giant
  L = lich (very dangerous undead)
  M = mummy
  N = naga
  O = ogre
  T = troll (regenerates -- eat corpse to gain regeneration)
  Z = zombie
  & = demon
  ; = sea creature
  : = lizard-like (newt, gecko)
  ' = golem
  @ = human (could be shopkeeper or guard -- do NOT attack shopkeepers!)

PET INDICATORS:
  Your pet appears as the same letter as its monster type but the language observation will say 'tame' (e.g., 'tame little dog').
"""

BOXOBAN_GAME_RULES = """\
ACTION FORMAT RULES -- Boxoban (Sokoban) Puzzle

VALID ACTIONS: north, south, east, west
These are the ONLY four valid actions. Diagonal moves are NOT supported.
Do NOT try: northeast, northwest, southeast, southwest, search, open, kick, fight, eat, etc.

GOAL: Push all boulders (`) onto fountain target tiles.
- To push a boulder, walk INTO it. The boulder moves in the direction you walked.
- You can only PUSH boulders, never pull them.
- A boulder against a wall CANNOT be pushed further in that direction.
- Once a boulder is on a fountain tile, it may still be pushed off -- be careful.
- Plan your moves! A carelessly pushed boulder can make the puzzle unsolvable.

IMPORTANT: '#' symbols on the map are IRON BARS -- you CANNOT walk through them.
They look like corridors but are actually impassable barriers.

STRATEGY:
- Before pushing, ask: will this boulder get stuck against a wall?
- Work on boulders furthest from walls first.
- Keep escape routes open so you can get behind boulders to push them.
- If your action has NO EFFECT (you don't move), that direction is blocked.

MAP SYMBOLS:
  @ = you (the player)
  . = floor (walkable)
  ` = boulder (push by walking into it)
  fountain/target = where boulders need to go
  - = wall (cannot pass)
  | = wall (cannot pass)
  # = iron bars (CANNOT pass -- looks like corridor but is blocked)
  (space) = rock/void (cannot pass)
"""

NLE_MEMORY_SYSTEM_PROMPT = """\
You are an agent playing NetHack. The following are the valid actions you can output, one at a time:

MOVEMENT: north, east, south, west, northeast, southeast, southwest, northwest (move exactly 1 tile). Also: far north, far east, far south, far west, far northeast, far southeast, far southwest, far northwest (sprint in that direction until hitting an obstacle -- wall, monster, item, or door).

STAIRS: up (go up, must be standing on '<'), down (go down, must be standing on '>').

UTILITY: wait, search, look, pickup, open (then direction), close (then direction), kick (then direction), fight (attack unseen monster, then direction), inventory.

RANGED: fire (shoots quivered ammo, then direction), throw (then item letter, then direction), zap (then wand letter, then direction for directional wands), quiver (select ammo, then item letter).

ITEM USAGE (all two-step -- output the command, then respond with the item letter when prompted; NEVER combine into one action): eat, quaff, wear, wield, read, apply, puton, remove, takeoff, takeoffall, drop, droptype, dip, rub, invoke, force, loot, offer, pay, swap, twoweapon, untrap, wipe.

SPELLS / SPECIAL: cast, engrave, enhance, annotate, call, jump, monster, teleport, overview, pray.

MENU / PROMPT RESPONSES: esc (exit menu), minus, space, apos, 0-9 (for responding to in-game prompts).

CRITICAL ACTION RULES:
- To attack a visible monster in MELEE: simply MOVE into its tile (e.g. output 'north' if monster is north). Do NOT use 'fight' for visible monsters.
- To attack at RANGE: first 'quiver' to select ammo, then 'fire' + direction. Or 'throw' + item letter + direction. Or 'zap' + wand letter + direction.
- Item commands (eat, quaff, wield, etc.) are ALWAYS two-step: first the command word (e.g. 'eat'), then the item letter (e.g. 'd') when the game prompts. 'eat d' as one action is INVALID.
- Items on the floor are auto-collected when you walk onto them. Gold is always instant. You do NOT need 'pickup' in normal play.
- When the game asks "In what direction?", respond with a direction word.
- When the game asks "What do you want to X? [abc or ?*]", respond with JUST the single item letter.
- Consult game_rules.txt in your memory files for detailed action format rules.
- Consult nethack_symbols.txt in your memory files when you see an unfamiliar symbol on the ASCII map.
- Both reference files can be updated -- APPEND new knowledge you discover.

READING THE OBSERVATION:
- The language observation describes entities around you with numeric step distances. '1 step away (directly reachable)' means you can reach it in exactly 1 move.
- The MAP ANALYSIS section (if present) shows passable/blocked directions computed programmatically. TRUST IT -- it is accurate.
- The ASCII map shows the dungeon layout. Find '@' (yourself) and study adjacent tiles before moving. Refer to nethack_symbols.txt for symbols.
- Read the status line every turn: watch HP, hunger status, and conditions. If Hungry or Weak, eat. If HP is low, consider retreat or healing.
- Check your dungeon level (Dlvl). At Dlvl:1, going UP exits the dungeon and ends the game. Always go DOWN to progress.

Your goal is to get as far as possible in the game.

PLAY!"""

MINIHACK_MEMORY_SYSTEM_PROMPT = """\
You are an agent playing MiniHack. The following are the valid actions you can output, one at a time:

MOVEMENT: north, east, south, west, northeast, southeast, southwest, northwest (move exactly 1 tile).

OTHER ACTIONS: open (open a door, game asks direction -- respond with direction word), kick (kick something, game asks direction), search (search for hidden doors/passages).

CRITICAL ACTION RULES:
- To attack a visible monster: simply MOVE into its tile. Do NOT try to combine actions.
- 'open' is two-step: first 'open', then the direction when prompted.
- Items on the floor are auto-collected when you walk onto them. Gold is always instant.
- Consult game_rules.txt in your memory files for detailed action format rules.
- Consult nethack_symbols.txt for map symbol meanings.
- Both files can be updated -- APPEND new knowledge you discover.

READING THE OBSERVATION:
- The MAP ANALYSIS section (if present) shows passable/blocked directions. TRUST IT -- it is accurate.
- The language observation uses numeric step distances: '1 step away (directly reachable)' means exactly 1 move away. Study these carefully before choosing a direction.
- The ASCII map shows the level. Find '@' (yourself). '<' is stairs up, '>' is stairs down, '-' and '|' are walls, '.' is floor, '#' is corridor. Do NOT move into walls or rock (spaces).
- Read the status line: watch HP and hunger.
- If your action has NO EFFECT (observation unchanged), try a DIFFERENT action.

Your goal is to explore the level and reach the stairs down.

PLAY!"""

BOXOBAN_MEMORY_SYSTEM_PROMPT = """\
You are an agent solving a Boxoban (Sokoban) puzzle in MiniHack.

VALID ACTIONS -- these are the ONLY actions you can output:
  north, south, east, west

NO OTHER ACTIONS EXIST. Diagonal moves (northeast, etc.), 'search', 'open', 'kick', and all other commands are INVALID and will be rejected.

PUZZLE RULES:
- Push boulders (`) by walking into them. The boulder slides one tile in your movement direction.
- You can only PUSH, never pull.
- A boulder against a wall cannot be pushed further that way.
- Goal: push ALL boulders onto the target (fountain) tiles.
- Consult game_rules.txt in your memory for strategy tips.

IMPORTANT: '#' symbols are IRON BARS -- you CANNOT walk through them!
They look like corridors but are blocked. Do NOT repeatedly try to move through '#'.

READING THE OBSERVATION:
- The MAP ANALYSIS section shows which directions you can move. TRUST IT -- it is accurate.
- The ASCII map shows the puzzle. '@' is you, '`' is a boulder, '-' and '|' are walls, '#' is bars.
- Think before you push! Can you still reach all sides of this boulder after pushing?
- Avoid pushing boulders into corners or against walls where they become stuck.
- If a move has NO EFFECT, that direction is BLOCKED. Try a different direction.

Your goal is to solve the puzzle by pushing all boulders onto the target tiles.

PLAY!"""

INITIAL_FILES_BY_ENV = {
    "crafter": {
        "map_notes.txt": "",
        "crafting_progress.txt": "",
        "survival_log.txt": "",
        "actions_log.txt": "",
        "strategy.txt": "",
    },
    "nle": {
        "game_rules.txt": NLE_GAME_RULES,
        "nethack_symbols.txt": NLE_SYMBOLS,
        "dungeon_map.txt": "",
        "monster_encounters.txt": "",
        "inventory.txt": "",
        "actions_log.txt": "",
        "strategy.txt": "",
    },
    "minihack": {
        "game_rules.txt": NLE_GAME_RULES,
        "nethack_symbols.txt": NLE_SYMBOLS,
        "dungeon_map.txt": "",
        "actions_log.txt": "",
        "inventory.txt": "",
        "strategy.txt": "",
    },
}

INITIAL_FILES_BOXOBAN = {
    "game_rules.txt": BOXOBAN_GAME_RULES,
    "nethack_symbols.txt": NLE_SYMBOLS,
    "dungeon_map.txt": "",
    "actions_log.txt": "",
    "strategy.txt": "",
}

INITIAL_FILES_DEFAULT = {
    "exploration.txt": "",
    "actions_log.txt": "",
    "inventory.txt": "",
    "strategy.txt": "",
}

SPATIAL_GUIDANCE = {
    "crafter": (
        "SPATIAL AWARENESS: Carefully read the terrain description around you. "
        "Track what surrounds you (water, lava, trees, stone, ore, etc.) and in "
        "which direction. In map_notes.txt, log your position and nearby features "
        "in a structured way, e.g.: 'Pos (3,5): N=tree, E=stone, S=water, W=grass'. "
        "Track what you have crafted and what you still need in crafting_progress.txt. "
        "Note: the 'Do' action interacts with what is directly in front of you -- if "
        "nothing interactable is there, 'Do' will have NO EFFECT.\n\n"
        "STATUS CHECK: Before choosing an action, read your status (health, food, "
        "drink, energy). If any resource is getting low, prioritize addressing it. "
        "Dying from starvation or thirst wastes all your progress."
    ),
    "nle": (
        "SPATIAL AWARENESS: The ASCII map shows the dungeon layout. Refer to "
        "nethack_symbols.txt in your memory for what each symbol means. '@' is "
        "you -- find it on the map and study what surrounds it in each direction. "
        "BEFORE choosing a direction, verify on the map that the destination tile "
        "is passable (floor '.', corridor '#', or open door). Do NOT move into "
        "walls ('-', '|') or solid rock (space). Record STATIC features (walls, "
        "stairs, doors, corridors) in dungeon_map.txt using structured format: "
        "'(x,y): floor | N:wall | E:door | S:corridor | W:dark'.\n\n"
        "DYNAMIC ENTITIES: Monsters and pets MOVE every turn. Do NOT record their "
        "positions as permanent landmarks. Rely on the CURRENT observation for "
        "monster/NPC locations.\n\n"
        "STATUS CHECK: Read the status line carefully every turn. Watch for "
        "Hunger (Hungry/Weak/Fainting -- eat immediately), low HP (consider "
        "retreat, healing, or prayer), and status conditions.\n\n"
        "ACTION FORMAT: Many actions are multi-step (command, then item/direction "
        "when prompted). Consult game_rules.txt in your memory if unsure about "
        "the correct format for any action. To attack a monster in melee, simply "
        "move into its tile."
    ),
    "minihack": (
        "SPATIAL AWARENESS: The ASCII map shows the level layout. Refer to "
        "nethack_symbols.txt in your memory for what each symbol means. '@' is "
        "you -- find it on the map and study what surrounds it. '<' is stairs up, "
        "'>' is stairs down. BEFORE choosing a direction, verify on the map that "
        "the target tile is passable -- do NOT move into '-' (wall), '|' (wall), "
        "or solid rock (space characters). Record STATIC features in dungeon_map.txt: "
        "'(x,y): floor | N:wall | E:stairs_up | S:corridor | W:dark'.\n\n"
        "DYNAMIC ENTITIES: Monsters and pets MOVE every turn. Do NOT record their "
        "positions as permanent landmarks. Rely on the CURRENT observation for "
        "monster/NPC locations.\n\n"
        "STATUS CHECK: Read the status line every turn. Watch for Hunger and "
        "low HP.\n\n"
        "ACTION FORMAT: Many actions are multi-step. Consult game_rules.txt in "
        "your memory if unsure. To attack a monster, simply move into its tile."
    ),
}

SPATIAL_GUIDANCE_DEFAULT = ("SPATIAL AWARENESS: Pay careful attention to spatial descriptions and any map or grid in the observation. Log positions and landmarks in a structured format so you can navigate effectively.")

LOG_SYSTEM = """\
You are playing a game and maintaining your external memory.

{action_description}

{diff_notice}

{spatial_guidance}

{map_analysis}

ENVIRONMENT RESPONSE TO YOUR PREVIOUS ACTION:
{observation}

YOUR MEMORY FILES:
{file_index}

AVAILABLE COMMANDS:
  <|SEARCH|>filename.txt|keyword<|END|>    Search a file for lines matching a keyword
  <|READ|>filename.txt<|END|>              Read an entire file
  <|TAIL|>filename.txt|N<|END|>            Read the last N lines of a file
  <|COUNT|>filename.txt|keyword<|END|>     Count lines matching a keyword
  <|APPEND|>filename.txt|Your note<|END|>  Add a line to a file
  <|WRITE|>filename.txt|Content<|END|>     Overwrite a file entirely
  <|CREATE|>new_file.txt<|END|>            Create a new empty file
  <|DONE|>                                 Finish logging, move on to action selection

IMPORTANT RULES:
1. BEFORE appending or writing to ANY file, first SEARCH or READ it to check whether the information is already recorded. Do NOT duplicate entries.
2. When logging, attribute the entry to the action that caused it. Format: "Step N action 'X': outcome description"
3. Not everything needs logging -- skip routine movements with no new information. But DO log:
   - Failed or invalid actions (e.g. "You can't go there")
   - UNCHANGED observations (the action had NO EFFECT -- this is important!)
   - Discoveries (new rooms, items, stairs, doors, monsters)
   - Inventory changes (picked up / lost items)
   - Damage taken or dealt
   - Unexpected outcomes
   - Important spatial information (coordinates, landmarks, obstacles)

You may issue multiple commands per turn. Finish with <|DONE|>."""

LOG_CONTINUE = """\
Command results:
{results}

Updated files:
{file_index}

Continue logging (remember: SEARCH before writing to avoid duplicates), or output <|DONE|> to move on to action selection."""

PLAN_SYSTEM = """\
{instruction_prompt}

{direction_prompt_notice}

You have an external memory system to help you make better decisions. Before committing to an action, you should search your memory for relevant information.

{task_tips}

{spatial_guidance}

{stuckness_warning}

{recent_actions_section}

{map_analysis}

{stairs_directive}

{exploration_guidance}

{mandatory_exploration}

YOUR MEMORY FILES:
{file_index}

RECENT ACTIONS LOG (auto-loaded):
{auto_memory_context}

CURRENT OBSERVATION:
{observation}

PROCESS -- follow these steps IN ORDER:
1. Read the observation. Study the ASCII map and identify your position ('@').
2. Check status (HP, hunger). Address urgent needs first (eat if hungry, fight if monster adjacent).
3. *** If there is a MANDATORY MOVEMENT DIRECTIVE above, you MUST follow it. Output that direction immediately. ***
4. *** If STAIRS DOWN '>' is adjacent, move toward it immediately to complete the level. ***
5. *** If GOAL DIRECTION shows stairs location, navigate persistently toward that direction. ***
   - If LAVA BARRIER is detected, follow the DETOUR instructions instead of going toward lava.
6. Check EXPLORATION STATUS. You MUST move toward UNVISITED tiles when available. Do NOT revisit tiles you have already been to.
7. If DOORS ON MAP shows a nearby door and you are stuck, navigate toward it and use 'open'.
8. If none of the above apply, think about what action to take.
9. BEFORE committing, SEARCH your memory for information related to your planned action:
   - Search actions_log.txt for your planned direction to check if it failed before.
   - Search game_rules.txt for correct action format if unsure.
10. When confident, commit: <|ACTION|>your_chosen_action<|END|>

AVAILABLE COMMANDS (read-only -- no writing in this phase):
  <|SEARCH|>filename.txt|keyword<|END|>    Search for lines matching keyword
  <|READ|>filename.txt<|END|>              Read entire file
  <|TAIL|>filename.txt|N<|END|>            Read last N lines of a file
  <|COUNT|>filename.txt|keyword<|END|>     Count matching lines
  <|ACTION|>your_chosen_action<|END|>      Commit to an action

You MUST eventually output exactly one <|ACTION|> to commit.
Think step by step: propose -> search -> validate -> commit or revise."""

PLAN_CONTINUE = """\
Search results:
{results}

Based on these results, decide:
- If your planned action looks safe, commit: <|ACTION|>your_action<|END|>
- If memory warns against it, revise your plan, search for the new action, then commit.

You can issue more SEARCH/READ/TAIL/COUNT commands, or commit with <|ACTION|>."""

PLAN_DIRECTION_ONLY = """\
{direction_prompt_notice}

{map_analysis}

CURRENT OBSERVATION:
{observation}

Choose a PASSABLE direction from the map analysis above.
You MUST output ONLY a direction word inside the ACTION tags.
Do NOT output any other command. Do NOT search memory.

<|ACTION|>your_direction_here<|END|>"""

_OPEN = {
    "READ":   r"<\|?READ\|?>",
    "SEARCH": r"<\|?SEARCH\|?>",
    "TAIL":   r"<\|?TAIL\|?>",
    "COUNT":  r"<\|?COUNT\|?>",
    "APPEND": r"<\|?APPEND\|?>",
    "WRITE":  r"<\|?WRITE\|?>",
    "CREATE": r"<\|?CREATE\|?>",
    "ACTION": r"<\|?ACTION\|?>",
    "DONE":   r"<\|?DONE\|?>",
}
_END = r"<\|?END\|?[>}]"

_ACTION_PATTERNS = [
    re.compile(r'<\|?ACTION\|?>(.*?)<\|?END\|?[>}]', re.DOTALL | re.IGNORECASE),
    re.compile(r'<ACTION>(.*?)</ACTION>', re.DOTALL | re.IGNORECASE),
    re.compile(r'(?:^|\n)\s*ACTION:\s*([^\n]{1,40})\s*(?:\n|$)', re.IGNORECASE),
]

class MemoryFileSystem:
    def __init__(self, root_dir, initial_files=None, max_search_results=MAX_SEARCH_RESULTS):
        self.root_dir = root_dir
        self.max_search_results = max_search_results  
        os.makedirs(self.root_dir, exist_ok=True)
        if initial_files:
            for name, content in initial_files.items():
                self._write_raw(self._safe_name(name), str(content))

    def _safe_name(self, filename):
        name = os.path.basename(filename.replace("\\", "/"))
        name = name.replace("..", "")
        if not name:
            name = "unnamed.txt"
        return name

    def _path(self, filename):
        return os.path.join(self.root_dir, filename)

    def _exists(self, filename):
        return os.path.isfile(self._path(filename))

    def _read_raw(self, filename):
        path = self._path(filename)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _write_raw(self, filename, content):
        with open(self._path(filename), "w", encoding="utf-8") as f:
            f.write(content)

    def _list_files(self):
        if not os.path.isdir(self.root_dir):
            return []
        return sorted(
            f for f in os.listdir(self.root_dir)
            if os.path.isfile(os.path.join(self.root_dir, f))
        )

    def read(self, filename):
        filename = self._safe_name(filename)
        if not self._exists(filename):
            return "[ERROR: file '{}' does not exist]".format(filename)
        content = self._read_raw(filename)
        if not content.strip():
            return "[FILE IS EMPTY]"
        return content

    def search(self, filename, keyword):
        filename = self._safe_name(filename)
        if not self._exists(filename):
            return "[ERROR: file '{}' does not exist]".format(filename)
        content = self._read_raw(filename)
        if not content.strip():
            return "[FILE IS EMPTY -- no matches]"
        matches = [
            line for line in content.splitlines()
            if keyword.lower() in line.lower()
        ]
        if not matches:
            return "[No lines matching '{}']".format(keyword)
        
        total = len(matches)
        if total > self.max_search_results:
            matches = matches[-self.max_search_results:]
            return "[Showing last {} of {} matches for '{}']\n{}".format(
                self.max_search_results, total, keyword, "\n".join(matches),
            )
        return "\n".join(matches)

    def tail(self, filename, n=10):
        filename = self._safe_name(filename)
        if not self._exists(filename):
            return "[ERROR: file '{}' does not exist]".format(filename)
        content = self._read_raw(filename)
        if not content.strip():
            return "[FILE IS EMPTY]"
        lines = content.splitlines()
        tail_lines = lines[-n:]
        return "\n".join(tail_lines)

    def count(self, filename, keyword):
        filename = self._safe_name(filename)
        if not self._exists(filename):
            return "[ERROR: file '{}' does not exist]".format(filename)
        content = self._read_raw(filename)
        if not content.strip():
            return "[0 matches for '{}' -- file is empty]".format(keyword)
        matches = [
            line for line in content.splitlines()
            if keyword.lower() in line.lower()
        ]
        return "[{} line(s) matching '{}']".format(len(matches), keyword)

    def append(self, filename, content):
        filename = self._safe_name(filename)
        path = self._path(filename)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content.rstrip("\n") + "\n")
        return "[Appended to {}]".format(filename)

    def write(self, filename, content):
        filename = self._safe_name(filename)
        if content.strip():
            self._write_raw(filename, content.rstrip("\n") + "\n")
        else:
            self._write_raw(filename, "")
        return "[Wrote to {}]".format(filename)

    def create(self, filename):
        filename = self._safe_name(filename)
        if self._exists(filename):
            return "[File '{}' already exists]".format(filename)
        self._write_raw(filename, "")
        return "[Created {}]".format(filename)

    def file_index(self):
        files = self._list_files()
        if not files:
            return "  (no files yet)"
        lines = []
        for name in files:
            content = self._read_raw(name) or ""
            n_lines = len(content.splitlines()) if content.strip() else 0
            lines.append("  - {} ({} lines)".format(name, n_lines))
        return "\n".join(lines)

    def snapshot(self):
        result = {}
        for name in self._list_files():
            result[name] = self._read_raw(name) or ""
        return result

    def reset(self, initial_files=None):
        import shutil
        if os.path.isdir(self.root_dir):
            shutil.rmtree(self.root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        if initial_files:
            for name, content in initial_files.items():
                self._write_raw(self._safe_name(name), str(content))

_DIRECTION_OFFSETS = {
    "north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1),
    "northeast": (-1, 1), "northwest": (-1, -1),
    "southeast": (1, 1), "southwest": (1, -1),
}
_TILE_PASSABLE = set(".#{_")
_TILE_PASSABLE_BOXOBAN = set(".{_")  
_TILE_WALL = set("-|")
_TILE_DANGEROUS = set("}^")
_TILE_ITEM = set("$)[!?/=\"(*%~(")

def _classify_tile(char, tile_passable=None):
    if tile_passable is None:
        tile_passable = _TILE_PASSABLE
    if char == ">":
        return "stairs_down"
    elif char == "<":
        return "stairs_up"
    elif char == "+":
        return "door"
    elif char in _TILE_WALL:
        return "wall"
    elif char == " ":
        return "rock"
    elif char in _TILE_DANGEROUS:
        return "dangerous"
    elif char in tile_passable:
        return "passable"
    elif char in _TILE_ITEM:
        return "item"
    elif char.isalpha() and char != "@":
        return "monster"
    elif char == "$":
        return "item"
    else:
        return "unknown"

def _offset_to_direction(row_offset, col_offset):
    parts = []
    if row_offset < 0:
        parts.append("north")
    elif row_offset > 0:
        parts.append("south")
    if col_offset > 0:
        parts.append("east")
    elif col_offset < 0:
        parts.append("west")
    return "".join(parts) if parts else "here"

def parse_nle_map(obs_text, task_name=None):
    if task_name and "boxoban" in task_name.lower():
        tile_passable = _TILE_PASSABLE_BOXOBAN
    else:
        tile_passable = _TILE_PASSABLE

    lines = obs_text.split("\n")
    player_row_idx = None
    player_col = None

    for i, line in enumerate(lines):
        at_col = line.find("@")
        if at_col < 0:
            continue
        stripped = line.rstrip()
        if len(stripped) < 3:
            continue
        alpha_count = sum(1 for c in stripped if c.isalpha() and c != "@")
        if len(stripped) > 0 and alpha_count / len(stripped) < 0.5:
            player_row_idx = i
            player_col = at_col
            break

    if player_row_idx is None:
        return ""

    map_rows = []
    player_map_row = 0
    for i in range(max(0, player_row_idx - 15), min(len(lines), player_row_idx + 15)):
        line = lines[i]
        stripped = line.rstrip()
        if not stripped:
            continue
        alpha_count = sum(1 for c in stripped if c.isalpha() and c != "@")
        total = max(len(stripped), 1)
        if alpha_count / total < 0.55 or i == player_row_idx:
            map_rows.append((i, stripped))

    for j, (orig_idx, _text) in enumerate(map_rows):
        if orig_idx == player_row_idx:
            player_map_row = j
            break

    analysis = []
    passable_dirs, blocked_dirs, dangerous_dirs = [], [], []
    stairs_dirs, monster_dirs, door_dirs = [], [], []

    for direction, (dr, dc) in _DIRECTION_OFFSETS.items():
        r = player_map_row + dr
        c = player_col + dc
        if 0 <= r < len(map_rows) and 0 <= c < len(map_rows[r][1]):
            char = map_rows[r][1][c]
            tile_type = _classify_tile(char, tile_passable)
            if tile_type == "stairs_down":
                stairs_dirs.append("{} -> '>' STAIRS DOWN -- go there!".format(direction))
                passable_dirs.append("{} (stairs down '>')".format(direction))
            elif tile_type == "stairs_up":
                stairs_dirs.append("{} -> '<' stairs up".format(direction))
                passable_dirs.append("{} (stairs up '<')".format(direction))
            elif tile_type == "door":
                door_dirs.append("{} -> '+' closed door (use 'open')".format(direction))
                passable_dirs.append("{} (door '+')".format(direction))
            elif tile_type in ("passable", "item"):
                passable_dirs.append("{} ('{}')".format(direction, char))
            elif tile_type == "wall":
                blocked_dirs.append("{} (wall '{}')".format(direction, char))
            elif tile_type == "rock":
                blocked_dirs.append("{} (rock/void)".format(direction))
            elif tile_type == "dangerous":
                dangerous_dirs.append("{} -> '{}' DANGEROUS".format(direction, char))
            elif tile_type == "monster":
                monster_dirs.append("{} -> '{}' monster (move into tile to attack)".format(direction, char))
                passable_dirs.append("{} (monster '{}')".format(direction, char))
            else:
                blocked_dirs.append("{} ('{}')".format(direction, char))
        else:
            blocked_dirs.append("{} (edge/void)".format(direction))

    stairs_locations = []
    door_locations = []
    for j, (_orig_idx, row_text) in enumerate(map_rows):
        for c_idx, ch in enumerate(row_text):
            if ch == ">":
                row_offset = j - player_map_row
                col_offset = c_idx - player_col
                dist = max(abs(row_offset), abs(col_offset))
                if dist > 1:
                    stairs_locations.append("'>' seen ~{} tiles {} on map".format(
                        dist, _offset_to_direction(row_offset, col_offset)))
            elif ch == "+":
                row_offset = j - player_map_row
                col_offset = c_idx - player_col
                dist = max(abs(row_offset), abs(col_offset))
                if dist > 1:
                    door_locations.append("'+' door ~{} tiles {}".format(
                        dist, _offset_to_direction(row_offset, col_offset)))

    analysis.append("=== MAP ANALYSIS ===")
    if stairs_dirs:
        analysis.append("*** STAIRS ADJACENT: " + "; ".join(stairs_dirs))
    if stairs_locations:
        analysis.append("*** STAIRS ON MAP: " + "; ".join(stairs_locations[:3]))
    if monster_dirs:
        analysis.append("MONSTERS ADJACENT: " + "; ".join(monster_dirs))
    if door_dirs:
        analysis.append("DOORS: " + "; ".join(door_dirs))
    if door_locations:
        analysis.append("DOORS ON MAP: " + "; ".join(door_locations[:4]))
    if dangerous_dirs:
        analysis.append("DANGER: " + "; ".join(dangerous_dirs))
    if passable_dirs:
        analysis.append("CAN MOVE: " + ", ".join(passable_dirs))
    if blocked_dirs:
        analysis.append("BLOCKED: " + ", ".join(blocked_dirs))
    return "\n".join(analysis)

def parse_log_calls(text):
    calls = []
    has_done = bool(re.search(_OPEN["DONE"], text))

    for m in re.finditer(_OPEN["READ"] + r"(.*?)" + _END, text, re.DOTALL):
        fn = m.group(1).strip()
        if fn:
            calls.append(("READ", {"filename": fn}))

    for m in re.finditer(_OPEN["SEARCH"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            calls.append(("SEARCH", {
                "filename": parts[0].strip(),
                "query": parts[1].strip(),
            }))

    for m in re.finditer(_OPEN["TAIL"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            try:
                n = int(parts[1].strip())
                calls.append(("TAIL", {"filename": parts[0].strip(), "n": n}))
            except ValueError:
                pass

    for m in re.finditer(_OPEN["COUNT"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            calls.append(("COUNT", {
                "filename": parts[0].strip(),
                "query": parts[1].strip(),
            }))

    for m in re.finditer(_OPEN["APPEND"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            calls.append(("APPEND", {
                "filename": parts[0].strip(),
                "content": parts[1].strip(),
            }))

    for m in re.finditer(_OPEN["WRITE"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip():
            calls.append(("WRITE", {
                "filename": parts[0].strip(),
                "content": parts[1].strip(),
            }))

    for m in re.finditer(_OPEN["CREATE"] + r"(.*?)" + _END, text, re.DOTALL):
        fn = m.group(1).strip()
        if fn:
            calls.append(("CREATE", {"filename": fn}))

    return calls, has_done

def parse_plan_calls(text):
    calls = []
    committed_action = None

    for m in re.finditer(_OPEN["READ"] + r"(.*?)" + _END, text, re.DOTALL):
        fn = m.group(1).strip()
        if fn:
            calls.append(("READ", {"filename": fn}))

    for m in re.finditer(_OPEN["SEARCH"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            calls.append(("SEARCH", {
                "filename": parts[0].strip(),
                "query": parts[1].strip(),
            }))

    for m in re.finditer(_OPEN["TAIL"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            try:
                n = int(parts[1].strip())
                calls.append(("TAIL", {"filename": parts[0].strip(), "n": n}))
            except ValueError:
                pass

    for m in re.finditer(_OPEN["COUNT"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            calls.append(("COUNT", {
                "filename": parts[0].strip(),
                "query": parts[1].strip(),
            }))

    for pattern in _ACTION_PATTERNS:
        m = pattern.search(text)
        if m:
            candidate = m.group(1).strip()
            if candidate and len(candidate) < 60:
                committed_action = candidate
                break

    return calls, committed_action

def _strip_action_tags(text):
    text = re.sub(r'<\|?ACTION\|?>.*?<\|?END\|?[>}]', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<ACTION>.*?</ACTION>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'(?:^|\n)\s*ACTION:\s*[^\n]{1,40}\s*(?=\n|$)', '', text, flags=re.IGNORECASE)
    return text.strip()

PLAN_HANDOFF = """\
Now commit your action.
You can issue more SEARCH/READ/TAIL/COUNT commands, or commit with <|ACTION|>your_action<|END|>."""

PLAN_HANDOFF_WITH_RESULTS = """\
Results:
{results}

Now commit your action.
You can issue more SEARCH/READ/TAIL/COUNT commands, or commit with <|ACTION|>your_action<|END|>."""

class MemoryAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder, config, gameplay_client_factory=None):
        super().__init__(client_factory, prompt_builder)
        self.config = config
        
        self.memory_client = self.client
        self.gameplay_client = gameplay_client_factory() if gameplay_client_factory else self.client
        self.max_log_iterations = OmegaConf.select(
            config, "agent.max_log_iterations", default=MAX_LOG_ITERATIONS,
        )
        self.max_validate_iterations = OmegaConf.select(
            config, "agent.max_validate_iterations", default=MAX_VALIDATE_ITERATIONS,
        )
        self.max_search_results = OmegaConf.select(
            config, "agent.max_search_results", default=MAX_SEARCH_RESULTS,
        )
        self.fallback_action_override = OmegaConf.select(
            config, "agent.fallback_action", default=None,
        )
        base_dir = OmegaConf.select(config, "agent.memory_dir", default=None)
        self._memory_base = base_dir or os.path.join(os.getcwd(), "balrog_memory")
        self._memory_dir = self._memory_base
        self._env_name = None
        self._task_name = None
        self.memory = None
        self.step_count = 0
        self.last_step_debug = {}
        self._prev_obs_text = None
        self._unchanged_count = 0
        self._action_tracker = ActionHistoryTracker()
        self._position_tracker = PositionTracker()
        self._stairs_navigator = StairsNavigator()
        logger.info("Memory agent files at: %s", self._memory_dir)

    def _get_initial_files(self):
        if self._task_name and "boxoban" in self._task_name.lower():
            return copy.deepcopy(INITIAL_FILES_BOXOBAN)
        
        if self._env_name and self._env_name in INITIAL_FILES_BY_ENV:
            return copy.deepcopy(INITIAL_FILES_BY_ENV[self._env_name])
        return copy.deepcopy(INITIAL_FILES_DEFAULT)

    def _get_spatial_guidance(self):
        if self._env_name and self._env_name in SPATIAL_GUIDANCE:
            return SPATIAL_GUIDANCE[self._env_name]
        return SPATIAL_GUIDANCE_DEFAULT

    def _get_instruction_prompt_override(self):
        if self._env_name == "nle":
            return NLE_MEMORY_SYSTEM_PROMPT
        elif self._env_name == "minihack":
            
            if self._task_name and "boxoban" in self._task_name.lower():
                return BOXOBAN_MEMORY_SYSTEM_PROMPT
            return MINIHACK_MEMORY_SYSTEM_PROMPT
        return None

    def _get_fallback_action(self):
        if self.fallback_action_override is not None:
            return self.fallback_action_override
        
        if self._task_name and "boxoban" in self._task_name.lower():
            return "north"
        if self._env_name and self._env_name in FALLBACK_ACTIONS:
            return FALLBACK_ACTIONS[self._env_name]
        return FALLBACK_ACTION_DEFAULT

    def _get_valid_directions(self):
        if self._task_name and "boxoban" in self._task_name.lower():
            return ["north", "south", "east", "west"]
        return list(_ALL_DIRECTIONS)

    def _get_auto_memory_context(self):
        if self.memory is None:
            return "(no memory initialized)"
        content = self.memory.tail("actions_log.txt", n=5)
        if not content or "[FILE IS EMPTY]" in content or "[ERROR:" in content:
            return "(no actions logged yet)"
        return content

    def configure_memory(self, env_name, task, episode_idx=0):
        self._env_name = env_name
        self._task_name = task
        self._memory_dir = os.path.join(
            self._memory_base, env_name, task, "ep_{:02d}".format(episode_idx),
        )
        self.memory = MemoryFileSystem(
            self._memory_dir, initial_files=self._get_initial_files(),
            max_search_results=self.max_search_results,
        )

    def reset(self):
        self.prompt_builder.reset()
        self.step_count = 0
        self.last_step_debug = {}
        self._prev_obs_text = None
        self._unchanged_count = 0
        self._action_tracker.reset()
        self._position_tracker.reset()
        self._stairs_navigator.reset()
        if self.memory is not None:
            self.memory.reset(initial_files=self._get_initial_files())

    def act(self, obs, prev_action=None, prev_validated_action=None):
        obs_text = self._format_obs(obs)
        spatial_guidance = self._get_spatial_guidance()

        override = self._get_instruction_prompt_override()
        if override is not None:
            instruction_prompt = override
        else:
            instruction_prompt = self.prompt_builder.system_prompt or ""

        game_prompt_active = _is_game_prompt_active(obs_text)

        
        if not game_prompt_active:
            self._position_tracker.update(obs_text)

        if prev_validated_action is not None:
            self._action_tracker.record(prev_validated_action)

        diff_notice = ""
        if self._prev_obs_text is not None:
            if obs_text.strip() == self._prev_obs_text.strip():
                self._unchanged_count += 1
                if self._unchanged_count >= 3:
                    diff_notice = ("WARNING: The observation has been UNCHANGED for {} consecutive actions. You are STUCK. Your recent actions have had NO EFFECT on the environment. You MUST try a COMPLETELY DIFFERENT approach -- do not repeat any of your recent actions.").format(self._unchanged_count)
                else:
                    diff_notice = ("WARNING: The observation is IDENTICAL to the previous step. Your previous action likely had NO EFFECT. This is important to log -- the action may be invalid or inapplicable in the current state.")
            else:
                self._unchanged_count = 0

        map_analysis = ""
        if self._env_name in ("nle", "minihack"):
            map_analysis = parse_nle_map(obs_text, task_name=self._task_name)

        self._stairs_navigator.update_from_map_analysis(map_analysis)

        task_tips = get_task_tips(self._task_name)

        valid_dirs = self._get_valid_directions()

        recent_actions_section = self._action_tracker.get_recent_actions_text()
        action_warnings = self._action_tracker.get_warnings(
            valid_directions=valid_dirs)
        if action_warnings:
            warning_text = "\n".join(action_warnings)
            if recent_actions_section:
                recent_actions_section = recent_actions_section + "\n\n" + warning_text
            else:
                recent_actions_section = warning_text

        passable_dirs = _extract_passable_directions(map_analysis)
        
        filtered_passable = [d for d in passable_dirs if d in valid_dirs]
        exploration_guidance = self._position_tracker.get_exploration_guidance(
            filtered_passable)

        stairs_directive = self._stairs_navigator.get_stairs_directive(
            passable_directions=filtered_passable,
            map_analysis_text=map_analysis)

        mandatory_dir = self._position_tracker.get_mandatory_direction(
            filtered_passable)
        mandatory_exploration = ""
        if mandatory_dir and not game_prompt_active:
            visit_count = self._position_tracker.visit_counts.get(
                self._position_tracker.current_pos, 0)
            mandatory_exploration = (
                "***************************************************\n"
                "*** MANDATORY MOVEMENT DIRECTIVE ***\n"
                "You have been at your current position {} times.\n"
                "You are critically STUCK in a loop.\n"
                "YOUR REQUIRED MOVE IS: {}\n"
                "Do NOT go any other direction. Move {} NOW.\n"
                "***************************************************"
            ).format(visit_count, mandatory_dir.upper(), mandatory_dir)

        direction_prompt_notice = _detect_game_prompt(obs_text)

        auto_memory_context = self._get_auto_memory_context()

        total_in = 0
        total_out = 0

        log_debug = None
        if prev_action is not None and self.step_count > 0:
            log_debug = self._log_phase(
                obs_text, prev_action, prev_validated_action,
                diff_notice, spatial_guidance, map_analysis,
            )
            total_in += log_debug["input_tokens"]
            total_out += log_debug["output_tokens"]

        plan_debug = self._plan_validate_phase(
            obs_text, instruction_prompt, diff_notice,
            spatial_guidance, map_analysis,
            prev_validated_action=prev_validated_action,
            task_tips=task_tips,
            recent_actions_section=recent_actions_section,
            exploration_guidance=exploration_guidance,
            direction_prompt_notice=direction_prompt_notice,
            auto_memory_context=auto_memory_context,
            stairs_directive=stairs_directive,
            mandatory_exploration=mandatory_exploration,
            game_prompt_active=game_prompt_active,
        )
        total_in += plan_debug["input_tokens"]
        total_out += plan_debug["output_tokens"]

        final_action = plan_debug["final_action"]

        repaired_action, was_repaired = repair_combined_action(final_action)
        if was_repaired:
            logger.info(
                "Action repaired: '%s' -> '%s' (split combined two-step action)",
                final_action, repaired_action,
            )
            final_action = repaired_action

        pos_info = None
        if self._position_tracker.current_pos:
            pos_info = {
                "current_pos": self._position_tracker.current_pos,
                "visit_count": self._position_tracker.visit_counts.get(
                    self._position_tracker.current_pos, 0),
                "total_unique": len(self._position_tracker.visit_counts),
                "total_steps": len(self._position_tracker.positions),
            }

        self.last_step_debug = {
            "step": self.step_count,
            "log_phase": log_debug,
            "plan_phase": plan_debug,
            "memory_snapshot": self.memory.snapshot(),
            "diff_notice": diff_notice,
            "unchanged_count": self._unchanged_count,
            "map_analysis": map_analysis,
            "recent_actions": list(self._action_tracker.actions),
            "action_warnings": action_warnings,
            "position_tracking": pos_info,
            "exploration_guidance": exploration_guidance,
            "stairs_directive": stairs_directive,
            "mandatory_exploration": mandatory_exploration,
        }

        final = LLMResponse(
            model_id=plan_debug.get("model_id", "unknown"),
            completion=final_action,
            stop_reason="stop",
            input_tokens=total_in,
            output_tokens=total_out,
            reasoning=None,
        )

        self._prev_obs_text = obs_text
        self.step_count += 1
        return final

    def _format_obs(self, obs):
        parts = []
        long_ctx = obs.get("text", {}).get("long_term_context", "")
        short_ctx = obs.get("text", {}).get("short_term_context", "")
        if long_ctx:
            parts.append(long_ctx)
        if short_ctx:
            parts.append(short_ctx)
        return "\n".join(parts) if parts else "(no observation)"

    def _execute_read_op(self, op, args):
        if op == "READ":
            return self.memory.read(args["filename"])
        elif op == "SEARCH":
            return self.memory.search(args["filename"], args["query"])
        elif op == "TAIL":
            return self.memory.tail(args["filename"], args.get("n", 10))
        elif op == "COUNT":
            return self.memory.count(args["filename"], args["query"])
        return "[Unknown read op: {}]".format(op)

    def _execute_write_op(self, op, args):
        if op == "READ":
            return self.memory.read(args["filename"])
        elif op == "SEARCH":
            return self.memory.search(args["filename"], args["query"])
        elif op == "TAIL":
            return self.memory.tail(args["filename"], args.get("n", 10))
        elif op == "COUNT":
            return self.memory.count(args["filename"], args["query"])
        elif op == "APPEND":
            return self.memory.append(args["filename"], args["content"])
        elif op == "WRITE":
            return self.memory.write(args["filename"], args["content"])
        elif op == "CREATE":
            return self.memory.create(args["filename"])
        return "[Unknown op: {}]".format(op)

    def _format_tool_result(self, op, args, result):
        if op == "READ":
            header = "READ {}".format(args["filename"])
        elif op == "SEARCH":
            header = "SEARCH {} for '{}'".format(
                args["filename"], args["query"],
            )
        elif op == "TAIL":
            header = "TAIL {} (last {} lines)".format(
                args["filename"], args.get("n", 10),
            )
        elif op == "COUNT":
            header = "COUNT {} for '{}'".format(
                args["filename"], args["query"],
            )
        elif op == "APPEND":
            header = "APPEND to {}".format(args["filename"])
            return header, "-- {} --\n{}".format(header, result)
        elif op == "WRITE":
            header = "WRITE to {}".format(args["filename"])
            return header, "-- {} --\n{}".format(header, result)
        elif op == "CREATE":
            header = "CREATE {}".format(args["filename"])
            return header, "-- {} --\n{}".format(header, result)
        else:
            return result, result
        return header, "-- {} --\n{}".format(header, result)

    def _log_phase(self, obs_text, prev_action, prev_validated_action,
                   diff_notice, spatial_guidance, map_analysis):
        prev_step = self.step_count - 1
        file_index = self.memory.file_index()

        if prev_validated_action is not None and prev_validated_action != prev_action:
            action_description = (
                "At the PREVIOUS step (step {prev_step}) your LLM output was: "
                "\"{prev_action}\"\n"
                "This was NOT a valid action. The environment executed the "
                "default action \"{prev_validated}\" instead.\n"
                "THIS IS AN IMPORTANT FAILURE TO LOG -- your output did not "
                "match any valid action."
            ).format(
                prev_step=prev_step,
                prev_action=prev_action,
                prev_validated=prev_validated_action,
            )
        else:
            effective = prev_validated_action or prev_action
            action_description = ("At the PREVIOUS step (step {prev_step}) you took action: \"{effective}\"").format(prev_step=prev_step, effective=effective)

        initial_prompt = LOG_SYSTEM.format(
            action_description=action_description,
            diff_notice=diff_notice,
            spatial_guidance=spatial_guidance,
            map_analysis=map_analysis,
            observation=obs_text,
            file_index=file_index,
        )

        messages = [Message(role="user", content=initial_prompt)]

        total_in = 0
        total_out = 0
        turns = []
        any_write_ops = False

        for iteration in range(self.max_log_iterations):
            current_prompt = messages[-1].content

            try:
                response = self.memory_client.generate(messages)
            except Exception as e:
                logger.error("Log phase LLM call failed: %s", e)
                turns.append({
                    "prompt": current_prompt,
                    "raw_response": "[ERROR: {}]".format(e),
                    "tool_results": "",
                })
                break

            total_in += response.input_tokens
            total_out += response.output_tokens
            assistant_text = response.completion

            calls, has_done = parse_log_calls(assistant_text)

            for op, _args in calls:
                if op in ("APPEND", "WRITE", "CREATE"):
                    any_write_ops = True

            result_parts = []
            for op, args in calls:
                result = self._execute_write_op(op, args)
                _header, formatted = self._format_tool_result(op, args, result)
                result_parts.append(formatted)

            results_text = "\n\n".join(result_parts) if result_parts else ""

            turns.append({
                "prompt": current_prompt,
                "raw_response": assistant_text,
                "tool_results": results_text,
            })

            messages.append(Message(role="assistant", content=assistant_text))

            if has_done:
                break
            if not calls:
                break

            updated_index = self.memory.file_index()
            continue_msg = LOG_CONTINUE.format(
                results=results_text,
                file_index=updated_index,
            )
            messages.append(Message(role="user", content=continue_msg))

        
        
        if not any_write_ops and prev_validated_action:
            effective = prev_validated_action or prev_action
            auto_entry = "Step {} action '{}': [auto-logged]".format(
                prev_step, effective)
            self.memory.append("actions_log.txt", auto_entry)

        return {
            "initial_prompt": initial_prompt,
            "turns": turns,
            "input_tokens": total_in,
            "output_tokens": total_out,
        }

    def _plan_validate_phase(self, obs_text, instruction_prompt,
                             diff_notice, spatial_guidance, map_analysis,
                             prev_validated_action=None,
                             task_tips="", recent_actions_section="",
                             exploration_guidance="",
                             direction_prompt_notice="",
                             auto_memory_context="",
                             stairs_directive="",
                             mandatory_exploration="",
                             game_prompt_active=False):
        file_index = self.memory.file_index()

        
        if game_prompt_active and direction_prompt_notice:
            initial_prompt = PLAN_DIRECTION_ONLY.format(
                direction_prompt_notice=direction_prompt_notice,
                map_analysis=map_analysis,
                observation=obs_text,
            )
        else:
            
            stuckness_warning = ""
            if diff_notice:
                stuckness_warning = diff_notice
                if prev_validated_action:
                    stuckness_warning += (
                        "\nYour last executed action was '{}'. "
                        "Do NOT repeat it -- choose something different."
                    ).format(prev_validated_action)

            initial_prompt = PLAN_SYSTEM.format(
                instruction_prompt=instruction_prompt,
                direction_prompt_notice=direction_prompt_notice,
                task_tips=task_tips,
                spatial_guidance=spatial_guidance,
                stuckness_warning=stuckness_warning,
                recent_actions_section=recent_actions_section,
                map_analysis=map_analysis,
                stairs_directive=stairs_directive,
                exploration_guidance=exploration_guidance,
                mandatory_exploration=mandatory_exploration,
                file_index=file_index,
                auto_memory_context=auto_memory_context,
                observation=obs_text,
            )

        messages = [Message(role="user", content=initial_prompt)]

        total_in = 0
        total_out = 0
        turns = []
        final_action = None
        model_id = "unknown"

        for iteration in range(self.max_validate_iterations):
            current_prompt = messages[-1].content

            try:
                response = self.memory_client.generate(messages)
            except Exception as e:
                logger.error("Plan memory-consult LLM call failed: %s", e)
                turns.append({
                    "prompt": current_prompt,
                    "raw_response": "[ERROR: {}]".format(e),
                    "tool_results": "",
                    "committed_action": None,
                    "client": "memory",
                })
                break

            model_id = response.model_id
            total_in += response.input_tokens
            total_out += response.output_tokens
            assistant_text = response.completion

            calls, committed_action = parse_plan_calls(assistant_text)

            result_parts = []
            for op, args in calls:
                result = self._execute_read_op(op, args)
                _header, formatted = self._format_tool_result(op, args, result)
                result_parts.append(formatted)

            results_text = "\n\n".join(result_parts) if result_parts else ""

            turns.append({
                "prompt": current_prompt,
                "raw_response": assistant_text,
                "tool_results": results_text,
                "committed_action": committed_action,
                "client": "memory",
            })

            if committed_action is not None:
                stripped = _strip_action_tags(assistant_text)
                if not stripped:
                    stripped = "(memory consultation complete)"
                messages.append(Message(role="assistant", content=stripped))
                if results_text:
                    messages.append(Message(role="user", content=PLAN_HANDOFF_WITH_RESULTS.format(results=results_text)))
                else:
                    messages.append(Message(role="user", content=PLAN_HANDOFF))
                break

            if not calls:
                messages.append(Message(role="assistant", content=assistant_text))
                messages.append(Message(role="user", content=PLAN_HANDOFF))
                break

            messages.append(Message(role="assistant", content=assistant_text))
            messages.append(Message(role="user", content=PLAN_CONTINUE.format(results=results_text)))

        for iteration in range(self.max_validate_iterations):
            current_prompt = messages[-1].content

            try:
                response = self.gameplay_client.generate(messages)
            except Exception as e:
                logger.error("Plan gameplay LLM call failed: %s", e)
                turns.append({
                    "prompt": current_prompt,
                    "raw_response": "[ERROR: {}]".format(e),
                    "tool_results": "",
                    "committed_action": None,
                    "client": "gameplay",
                })
                break

            model_id = response.model_id
            total_in += response.input_tokens
            total_out += response.output_tokens
            assistant_text = response.completion

            calls, committed_action = parse_plan_calls(assistant_text)

            result_parts = []
            for op, args in calls:
                result = self._execute_read_op(op, args)
                _header, formatted = self._format_tool_result(op, args, result)
                result_parts.append(formatted)

            results_text = "\n\n".join(result_parts) if result_parts else ""

            turns.append({
                "prompt": current_prompt,
                "raw_response": assistant_text,
                "tool_results": results_text,
                "committed_action": committed_action,
                "client": "gameplay",
            })

            messages.append(Message(role="assistant", content=assistant_text))

            if committed_action is not None:
                final_action = committed_action
                break

            if not calls:
                logger.warning(
                    "Plan phase: no SEARCH/READ/TAIL/COUNT/ACTION found. Raw: %s", assistant_text[:200],
                )
                final_action = self._get_fallback_action()
                break

            continue_msg = PLAN_CONTINUE.format(results=results_text)
            messages.append(Message(role="user", content=continue_msg))

        if final_action is None:
            fallback = self._get_fallback_action()
            final_action = fallback
            logger.warning(
                "Plan phase exhausted %d iterations without <|ACTION|>. Defaulting to '%s'.", self.max_validate_iterations,
                fallback,
            )

        return {
            "initial_prompt": initial_prompt,
            "turns": turns,
            "final_action": final_action,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "model_id": model_id,
        }
