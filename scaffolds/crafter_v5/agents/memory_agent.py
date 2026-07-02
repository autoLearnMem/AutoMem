import copy
import logging
import os
import re

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

_CRAFTER_CRAFT_ACTIONS = frozenset([
    "Make Wood Pickaxe", "Make Stone Pickaxe", "Make Iron Pickaxe",
    "Make Wood Sword", "Make Stone Sword", "Make Iron Sword",
])

_CRAFTER_PLACE_ACTIONS = frozenset([
    "Place Table", "Place Stone", "Place Furnace", "Place Plant",
])

_CRAFTER_HOSTILE_MOBS = frozenset(["zombie", "skeleton"])

_CRAFTER_MATERIAL_REQUIREMENTS = {
    "Place Table": {"wood": 2},
    "Place Stone": {"stone": 1},
    "Place Furnace": {"stone": 4},
    "Place Plant": {"sapling": 1},
    "Make Wood Pickaxe": {"wood": 1},
    "Make Wood Sword": {"wood": 1},
    "Make Stone Pickaxe": {"wood": 1, "stone": 1},
    "Make Stone Sword": {"wood": 1, "stone": 1},
    "Make Iron Pickaxe": {"wood": 1, "coal": 1, "iron": 1},
    "Make Iron Sword": {"wood": 1, "coal": 1, "iron": 1},
}

def repair_combined_action(raw_action):
    stripped = raw_action.strip()
    parts = stripped.split(None, 1)
    if len(parts) < 2:
        return stripped, False
    cmd = parts[0].lower()
    if cmd == "far" and parts[1].strip().lower() in _ALL_DIRECTIONS:
        return stripped, False
    if cmd in _TWO_STEP_COMMANDS:
        return cmd, True
    return stripped, False

def repair_crafter_action(raw_action):
    stripped = raw_action.strip()
    lower = stripped.lower()

    diagonal_fixes = {
        "move north-east": "Move North", "move northeast": "Move North",
        "move north-west": "Move North", "move northwest": "Move North",
        "move south-east": "Move South", "move southeast": "Move South",
        "move south-west": "Move South", "move southwest": "Move South",
        "move north east": "Move North", "move north west": "Move North",
        "move south east": "Move South", "move south west": "Move South",
    }
    if lower in diagonal_fixes:
        return diagonal_fixes[lower], True

    valid_actions = {
        "noop": "Noop", "move west": "Move West", "move east": "Move East",
        "move north": "Move North", "move south": "Move South",
        "do": "Do", "sleep": "Sleep",
        "place table": "Place Table", "place stone": "Place Stone",
        "place furnace": "Place Furnace", "place plant": "Place Plant",
        "make wood pickaxe": "Make Wood Pickaxe",
        "make stone pickaxe": "Make Stone Pickaxe",
        "make iron pickaxe": "Make Iron Pickaxe",
        "make wood sword": "Make Wood Sword",
        "make stone sword": "Make Stone Sword",
        "make iron sword": "Make Iron Sword",
    }
    if lower in valid_actions and valid_actions[lower] != stripped:
        return valid_actions[lower], True

    return stripped, False

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
  # = corridor (walkable)
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
- Repeating the same action when the observation does not change is pointless -- try something different.

Your goal is to explore the level and reach the stairs down.

PLAY!"""

CRAFTER_MEMORY_SYSTEM_PROMPT = """\
You are an agent playing Crafter, a 2D survival and crafting game.

VALID ACTIONS (output exactly one of these):
  Noop: do nothing
  Move West / Move East / Move North / Move South: move on flat ground
  Do: interact with what is DIRECTLY IN FRONT of you (collect, drink, attack)
  Sleep: rest to restore energy (cannot act while sleeping)
  Place Table: place a crafting table (needs 2 wood, must face empty grass tile)
  Place Stone: place stone (needs stone, must face empty grass tile)
  Place Furnace: place a furnace (needs 4 stone, must face empty grass tile)
  Place Plant: place a plant (needs sapling, must face empty grass tile)
  Make Wood Pickaxe: craft (needs nearby table + 1 wood)
  Make Stone Pickaxe: craft (needs nearby table + 1 wood + 1 stone)
  Make Iron Pickaxe: craft (needs nearby table + furnace + 1 wood + 1 coal + 1 iron)
  Make Wood Sword: craft (needs nearby table + 1 wood)
  Make Stone Sword: craft (needs nearby table + 1 wood + 1 stone)
  Make Iron Sword: craft (needs nearby table + furnace + 1 wood + 1 coal + 1 iron)

CRITICAL RULES:
- "Do" targets what is directly in front of you. If nothing interactable is there, NO EFFECT.
- Mining stone/coal REQUIRES a Wood Pickaxe or better. Without one, "Do" does NOTHING.
- Mining iron REQUIRES a Stone Pickaxe or better. Diamond REQUIRES Iron Pickaxe.
- Placement REQUIRES facing an empty grass tile. Facing stone/water/tree/table/path FAILS.
  If placement fails, MOVE until you face grass, then try again.
- Crafting REQUIRES being ADJACENT to a placed table (and furnace for iron-tier items).
  If crafting has no effect, you are NOT close enough to a table -- MOVE to find one first.
- CHECK YOUR INVENTORY before crafting/placing. Each action costs materials:
  Place Table = 2 wood. Place Furnace = 4 stone. Crafting = 1 wood + tier materials.
- READ game_knowledge.txt in your memory for the full tech tree and prerequisites.

ACHIEVEMENTS (22 total -- unlock as many as possible):
  Collect Wood, Place Table, Eat Cow, Collect Sapling, Collect Drink,
  Make Wood Pickaxe, Make Wood Sword, Place Plant, Defeat Zombie,
  Collect Stone, Place Stone, Eat Plant, Defeat Skeleton,
  Make Stone Pickaxe, Make Stone Sword, Wake Up, Place Furnace,
  Collect Coal, Collect Iron, Make Iron Pickaxe, Make Iron Sword, Collect Diamond

SURVIVAL: Health/food/drink/energy are each 0-9.
  If food or drink hits 0, your health drops each step. Health 0 = death = game over.
  ALWAYS prioritize food/drink when they drop below 3.

PLAY!"""

CRAFTER_GAME_KNOWLEDGE = """\
=== CRAFTER TECH TREE & CRAFTING PREREQUISITES ===

PROGRESSION ORDER (follow this for maximum achievements):

TIER 1 -- NO TOOLS NEEDED:
  Collect Wood: Face a tree, use "Do". Gives wood and sometimes sapling.
  Collect Sapling: Face a tree, use "Do". Sometimes drops a sapling.
  Collect Drink: Face water/lake, use "Do". Restores drink.
  Eat Cow: Attack a cow with "Do" (having a sword helps). Restores food.
  Eat Plant: Face a grown plant, use "Do". Restores food.

TIER 2 -- NEEDS WOOD IN INVENTORY:
  Place Table: Face an EMPTY GRASS tile, use "Place Table". Costs 2 wood.
    WARNING: Facing stone/tree/water/path/table will FAIL. Turn until you face grass.
    IMPORTANT: Collect AT LEAST 3 wood BEFORE placing a table!
    You need 2 wood for table + 1 for pickaxe = 3 minimum. 4 for pickaxe+sword.

TIER 3 -- NEEDS NEARBY TABLE + WOOD:
  Make Wood Pickaxe: Use "Make Wood Pickaxe" while ADJACENT to a placed table. Costs 1 wood.
  Make Wood Sword: Use "Make Wood Sword" while ADJACENT to a placed table. Costs 1 wood.
  >>> After placing a table, IMMEDIATELY craft tools before walking away! <<<
  >>> If crafting fails, you are NOT adjacent to a table. Move to find one. <<<

TIER 4 -- NEEDS WOOD PICKAXE OR BETTER:
  Collect Stone: Face stone, use "Do". REQUIRES Wood Pickaxe minimum.
  Collect Coal: Face coal, use "Do". REQUIRES Wood Pickaxe minimum.
  >>> WITHOUT a pickaxe, "Do" on stone/coal does ABSOLUTELY NOTHING. <<<

TIER 5 -- NEEDS TABLE + WOOD + STONE:
  Make Stone Pickaxe: Near table, use "Make Stone Pickaxe". Costs 1 wood + 1 stone.
  Make Stone Sword: Near table, use "Make Stone Sword". Costs 1 wood + 1 stone.
  Place Stone: Face empty grass tile, use "Place Stone". Costs 1 stone.
  Place Furnace: Face empty grass tile, use "Place Furnace". Costs 4 stone.

TIER 6 -- NEEDS STONE PICKAXE OR BETTER:
  Collect Iron: Face iron ore, use "Do". REQUIRES Stone Pickaxe minimum.

TIER 7 -- NEEDS TABLE + FURNACE + WOOD + COAL + IRON:
  Make Iron Pickaxe: Near table AND furnace, use "Make Iron Pickaxe". Costs 1 wood + 1 coal + 1 iron.
  Make Iron Sword: Near table AND furnace, use "Make Iron Sword". Costs 1 wood + 1 coal + 1 iron.

TIER 8 -- NEEDS IRON PICKAXE:
  Collect Diamond: Face diamond, use "Do". REQUIRES Iron Pickaxe.

RECOMMENDED EARLY GAME SEQUENCE:
  1. Collect wood (Do on tree) -- repeat until you have 4+ wood
  2. Face grass/path, Place Table (costs 2 wood, leaves you 2)
  3. Make Wood Pickaxe IMMEDIATELY (costs 1 wood, stay near table!)
  4. Make Wood Sword IMMEDIATELY (costs 1 wood, stay near table!)
  5. Collect Drink (Do on water)
  6. Find and mine stone (Do on stone -- now works with pickaxe)
  7. Find and mine coal (Do on coal)
  8. Go back to table area, Place Furnace nearby (costs 4 stone!)
  9. Make Stone Pickaxe, Make Stone Sword
  10. Continue to iron tier...

SURVIVAL PRIORITIES:
  - Food/Drink below 3/9: STOP everything, find food/water NOW
  - Health below 3/9: Avoid combat, seek food/drink to heal
  - Energy low: Use "Sleep" to rest (you cannot act while sleeping)
  - Food and drink deplete over time. When either hits 0, health drops each step.

PLACEMENT RULES:
  All placement actions (Place Table/Stone/Furnace/Plant) require facing
  an EMPTY GRASS tile. If the tile in front has stone/tree/water/any object/path,
  placement FAILS. Move and turn until "You face grass" appears in the observation,
  THEN place.

COMBAT:
  - Face creature, use "Do" to attack.
  - Wood sword < Stone sword < Iron sword (more damage).
  - Zombies and skeletons are hostile.
  - Cows are passive food sources.
"""

CRAFTER_INITIAL_GOALS = """\
CURRENT GOAL: Collect wood (face a tree, use Do) -- collect 4+ before placing table
NEXT GOALS: Place table (2 wood) -> IMMEDIATELY Make wood pickaxe (1 wood) -> Make wood sword (1 wood)
PREREQUISITES FOR CURRENT GOAL: None -- just find and face a tree
ACHIEVEMENTS UNLOCKED: (none yet)
RECENT FAILURES: (none yet)
"""

CRAFTER_INITIAL_PROGRESS = """\
=== ACHIEVEMENT TRACKER ===
(Update with [X] when you unlock an achievement)
[ ] Collect Wood
[ ] Place Table
[ ] Make Wood Pickaxe
[ ] Make Wood Sword
[ ] Collect Drink
[ ] Eat Cow
[ ] Collect Sapling
[ ] Place Plant
[ ] Collect Stone
[ ] Collect Coal
[ ] Place Stone
[ ] Place Furnace
[ ] Make Stone Pickaxe
[ ] Make Stone Sword
[ ] Eat Plant
[ ] Defeat Zombie
[ ] Defeat Skeleton
[ ] Wake Up
[ ] Collect Iron
[ ] Make Iron Pickaxe
[ ] Make Iron Sword
[ ] Collect Diamond
"""

CRAFTER_LOG_INSTRUCTIONS = """\
CRAFTER-SPECIFIC LOGGING:
4. EVERY TURN, use WRITE on goals.txt to update your current state:
   - What is your current goal?
   - What are your next goals?
   - What prerequisites are you missing?
   - What achievements have you unlocked?
   - What recent failures should you remember?
   Use WRITE (not APPEND) so goals.txt always shows current state only.
5. When you UNLOCK a new achievement (first time collecting an item,
   first time placing/crafting something, defeating an enemy), update
   progress.txt: change [ ] to [X] for that achievement using WRITE.
6. When you PLACE a table or furnace, note the step number in
   actions_log.txt so you can find it later.
7. If "Do" had NO EFFECT, write WHY in actions_log.txt (e.g., "no pickaxe",
   "facing wrong tile", "nothing interactable in front").
8. If crafting (Make ...) had NO EFFECT, log that you were NOT near a table.
   You MUST move to find a table or place a new one before trying again.
9. Keep goals.txt, observations.txt, and actions_log.txt under 25 lines.
   When a file grows large, use WRITE to summarize and keep only the
   most important recent information."""

CRAFTER_PLAN_INSTRUCTIONS = """\
CRAFTER PLANNING RULES:
- VALID MOVES: Move North, Move East, Move South, Move West ONLY.
  Diagonal moves like 'Move North-East' do NOT exist and will be wasted.
- Before mining (Do on stone/coal/iron/diamond): CHECK inventory for required pickaxe.
- Before crafting: CHECK INVENTORY for materials AND verify you are ADJACENT to a table.
  Place Table costs 2 wood. Make Wood Pickaxe/Sword costs 1 wood each.
  Make Stone Pickaxe/Sword costs 1 wood + 1 stone. Place Furnace costs 4 stone.
  If you don't have enough materials, COLLECT MORE first -- do not attempt craft/place!
- Before placing: Verify you face GRASS (not stone, tree, water, path, or any object).
- If food or drink is at 3/9 or below: PRIORITIZE survival over all other goals.
- If "Do" keeps failing: READ game_knowledge.txt for prerequisites you may be missing.
- Follow the tech tree order -- don't skip tiers."""

INITIAL_FILES_BY_ENV = {
    "crafter": {
        "game_knowledge.txt": CRAFTER_GAME_KNOWLEDGE,
        "goals.txt": CRAFTER_INITIAL_GOALS,
        "progress.txt": CRAFTER_INITIAL_PROGRESS,
        "observations.txt": "",
        "actions_log.txt": "",
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

INITIAL_FILES_DEFAULT = {
    "exploration.txt": "",
    "actions_log.txt": "",
    "inventory.txt": "",
    "strategy.txt": "",
}

SPATIAL_GUIDANCE = {
    "crafter": (
        "SPATIAL AWARENESS: Read the terrain description carefully. Note what "
        "you face directly in front -- 'Do' only interacts with what is in front. "
        "If you need to interact with something to your north, first 'Move North' "
        "to turn and face it, then 'Do'. Only log truly NEW spatial discoveries "
        "to observations.txt (new resource locations, dangers).\n\n"
        "STATUS CHECK: Before choosing an action, read your status. If health, "
        "food, or drink is 3/9 or below, PRIORITIZE finding food (cow/plant) or "
        "water. Dying wastes ALL your progress.\n\n"
        "INVENTORY CHECK: Before mining or crafting, check your inventory. "
        "You need a Wood Pickaxe to mine stone/coal. You need a Stone Pickaxe "
        "to mine iron. You need an Iron Pickaxe for diamond. Without the right "
        "tool, 'Do' on these materials does NOTHING."
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

{inventory_changes}

{craft_failure_notice}

{threat_warning}

{vital_urgency}

{spatial_guidance}

{map_analysis}

ENVIRONMENT RESPONSE TO YOUR PREVIOUS ACTION:
{observation}

YOUR MEMORY FILES:
{file_index}

{compaction_warnings}

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

{env_specific_logging}

You may issue multiple commands per turn. Finish with <|DONE|>."""

LOG_CONTINUE = """\
Command results:
{results}

Updated files:
{file_index}

Continue logging (remember: SEARCH before writing to avoid duplicates), or output <|DONE|> to move on to action selection."""

PLAN_SYSTEM = """\
{instruction_prompt}

You have an external memory system to help you make better decisions. Before committing to an action, you should search your memory for relevant information.

{spatial_guidance}

{stuckness_warning}

{vital_urgency}

{threat_warning}

{repetition_warning}

{craft_context}

{tech_tree_status}

{stagnation_warning}

{noop_warning}

{map_analysis}

YOUR CURRENT GOALS:
{goals_content}

YOUR MEMORY FILES:
{file_index}

CURRENT OBSERVATION:
{observation}

PROCESS -- follow these steps:
1. Carefully read the observation. Check what is in front of you and what is around you.
2. Check your status (health, food, drink, energy) -- address urgent survival needs FIRST.
3. Review your CURRENT GOALS above. Choose an action that progresses toward your goal.
4. BEFORE committing, SEARCH your memory for information related to your planned action:
   - If you plan to mine or collect, search game_knowledge.txt for required tools.
   - If you plan to craft, verify you have the materials and are near a table.
   - If you plan to place something, verify you face an empty tile (grass/path).
   - If you see an unfamiliar symbol, search for it in your reference files.
5. Review what memory tells you. If it reveals your planned action is problematic
   (missing prerequisite, failed before, wrong format), REVISE your plan.
6. When you are confident, commit with:
   <|ACTION|>your_chosen_action<|END|>

{env_specific_planning}

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
_TILE_WALL = set("-|")
_TILE_DANGEROUS = set("}^")
_TILE_ITEM = set("$)[!?/=\"(*%~(")

def _classify_tile(char):
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
    elif char in _TILE_PASSABLE:
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

def parse_nle_map(obs_text):
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
            tile_type = _classify_tile(char)
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
    for j, (_orig_idx, row_text) in enumerate(map_rows):
        for c_idx, ch in enumerate(row_text):
            if ch == ">":
                row_offset = j - player_map_row
                col_offset = c_idx - player_col
                dist = max(abs(row_offset), abs(col_offset))
                if dist > 1:
                    stairs_locations.append("'>' seen ~{} tiles {} on map".format(
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

class MemoryAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder)
        self.config = config
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
        self.memory = None
        self.step_count = 0
        self.last_step_debug = {}
        self._prev_obs_text = None
        self._unchanged_count = 0
        
        self._recent_actions = []  
        
        self._prev_inventory_items = {}
        
        self._consecutive_craft_failures = 0
        
        self._prev_health = None
        
        self._completed_actions = set()
        
        self._consecutive_noops = 0
        
        self._steps_since_progress = 0
        
        self._known_inventory_types = set()
        logger.info("Memory agent files at: %s", self._memory_dir)

    def _get_initial_files(self):
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
            return MINIHACK_MEMORY_SYSTEM_PROMPT
        elif self._env_name == "crafter":
            return CRAFTER_MEMORY_SYSTEM_PROMPT
        return None

    def _get_fallback_action(self):
        if self.fallback_action_override is not None:
            return self.fallback_action_override
        if self._env_name and self._env_name in FALLBACK_ACTIONS:
            return FALLBACK_ACTIONS[self._env_name]
        return FALLBACK_ACTION_DEFAULT

    def _get_env_specific_logging(self):
        if self._env_name == "crafter":
            return CRAFTER_LOG_INSTRUCTIONS
        return ""

    def _get_env_specific_planning(self):
        if self._env_name == "crafter":
            return CRAFTER_PLAN_INSTRUCTIONS
        return ""

    def _parse_crafter_vitals(self, obs_text):
        vitals = {}
        for vital in ['health', 'food', 'drink', 'energy']:
            m = re.search(r'{}: (\d+)/9'.format(vital), obs_text)
            if m:
                vitals[vital] = int(m.group(1))
        return vitals

    def _get_vital_urgency(self, obs_text):
        if self._env_name != "crafter":
            return ""
        vitals = self._parse_crafter_vitals(obs_text)
        if not vitals:
            return ""

        warnings = []
        health = vitals.get('health', 9)
        food = vitals.get('food', 9)
        drink = vitals.get('drink', 9)

        if health <= 2:
            warnings.append(
                "CRITICAL: Health is {}/9! You are about to DIE. Find food (cow/plant) or drink (water) IMMEDIATELY.".format(health)
            )
        elif health <= 4:
            warnings.append(
                "WARNING: Health is {}/9. Avoid combat, find food or water.".format(health)
            )

        if food <= 1:
            warnings.append(
                "CRITICAL: Food is {}/9! Starvation imminent. Attack a cow (Do) or eat a plant NOW.".format(food)
            )
        elif food <= 3:
            warnings.append(
                "WARNING: Food is {}/9 and dropping. Find food soon.".format(food)
            )

        if drink <= 1:
            warnings.append(
                "CRITICAL: Drink is {}/9! Dehydration imminent. Face water and use Do NOW.".format(drink)
            )
        elif drink <= 3:
            warnings.append(
                "WARNING: Drink is {}/9 and dropping. Find water soon.".format(drink)
            )

        if not warnings:
            return ""
        return "=== SURVIVAL URGENCY ===\n" + "\n".join(warnings)

    def _parse_crafter_inventory_items(self, obs_text):
        inventory = {}
        in_inventory = False
        for line in obs_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("Your inventory:"):
                in_inventory = True
                continue
            if in_inventory:
                if stripped.startswith("- "):
                    m = re.match(r"- (.+?):\s*(\d+)", stripped)
                    if m:
                        inventory[m.group(1).strip()] = int(m.group(2))
                elif stripped:
                    
                    break
        return inventory

    def _get_inventory_changes(self, obs_text):
        if self._env_name != "crafter":
            return ""
        current = self._parse_crafter_inventory_items(obs_text)
        if not self._prev_inventory_items:
            return ""

        changes = []
        all_keys = set(current.keys()) | set(self._prev_inventory_items.keys())
        for key in sorted(all_keys):
            old_val = self._prev_inventory_items.get(key, 0)
            new_val = current.get(key, 0)
            if new_val > old_val:
                changes.append("GAINED: +{} {} (now {})".format(
                    new_val - old_val, key, new_val))
            elif new_val < old_val:
                changes.append("LOST: -{} {} (now {})".format(
                    old_val - new_val, key, new_val))

        if not changes:
            return ""
        return (
            "=== INVENTORY CHANGES (from previous action) ===\n"
            + "\n".join(changes)
            + "\nThese changes CONFIRM your previous action had an effect. "
            "Log accurately based on what actually changed."
        )

    def _get_repetition_warning(self):
        if not self._recent_actions:
            return ""

        recent = self._recent_actions[-8:]
        if not recent:
            return ""

        last_action = recent[-1][0]

        same_action_entries = [(a, e) for a, e in recent if a == last_action]
        fail_count = sum(1 for _, e in same_action_entries if not e)
        total_same = len(same_action_entries)

        if fail_count >= 3:
            
            if self._env_name == "crafter" and last_action in _CRAFTER_CRAFT_ACTIONS:
                return (
                    "=== CRAFT FAILURE ALERT ===\n"
                    "Action '{}' has FAILED {} times! You are NOT near a crafting table. "
                    "STOP trying to craft. You must FIRST either:\n"
                    "  1. Move around to find your previously placed table, OR\n"
                    "  2. Collect wood and place a NEW table (face grass, Place Table)\n"
                    "Only attempt crafting again when you are ADJACENT to a table."
                    .format(last_action, fail_count)
                )
            return (
                "=== REPETITION ALERT ===\n"
                "Action '{}' has had NO EFFECT {} out of {} recent attempts! "
                "This action is NOT WORKING. You MUST try something completely "
                "different. READ game_knowledge.txt to check if you are missing "
                "a prerequisite (e.g., pickaxe for mining, table for crafting, "
                "empty tile for placing).".format(last_action, fail_count, total_same)
            )
        elif total_same >= 5 and fail_count >= 2:
            return (
                "=== REPETITION WARNING ===\n"
                "Action '{}' repeated {} times with {} failures. "
                "Consider a different approach.".format(last_action, total_same, fail_count)
            )

        
        if len(self._recent_actions) >= 6:
            last_6 = [a for a, _ in self._recent_actions[-6:]]
            unique_last_6 = set(last_6)
            if len(unique_last_6) <= 2:
                move_count = sum(1 for a in last_6
                                 if a.startswith("Move"))
                if move_count >= 5:
                    return (
                        "=== STUCK WARNING ===\n"
                        "You have been moving back and forth ({}) for the last "
                        "6 actions with no progress. You are STUCK. Try exploring "
                        "in a completely NEW direction you haven't tried recently."
                        .format(", ".join(last_6))
                    )

        return ""

    def _get_goals_content(self):
        if self.memory is None:
            return "(no goals file)"
        content = self.memory.read("goals.txt")
        if content.startswith("[ERROR") or content == "[FILE IS EMPTY]":
            return "(no goals set -- update goals.txt in the LOG phase)"
        return content

    def _get_compaction_warnings(self):
        if self.memory is None:
            return ""
        warnings = []
        for name in self.memory._list_files():
            if name in ("game_knowledge.txt", "progress.txt"):
                continue  
            content = self.memory._read_raw(name)
            if content:
                lines = len(content.splitlines())
                if lines > 25:
                    warnings.append(
                        "FILE TOO LARGE: {} has {} lines. Use WRITE to summarize and keep only the most important recent information.".format(name, lines)
                    )
        if not warnings:
            return ""
        return "=== MEMORY MAINTENANCE ===\n" + "\n".join(warnings)

    def _get_craft_failure_notice(self, prev_validated_action, had_effect):
        if self._env_name != "crafter":
            return ""
        if prev_validated_action is None:
            return ""
        if prev_validated_action not in _CRAFTER_CRAFT_ACTIONS:
            return ""
        if had_effect:
            
            self._consecutive_craft_failures = 0
            return ""

        self._consecutive_craft_failures += 1
        if self._consecutive_craft_failures >= 2:
            return (
                "=== CRAFTING FAILED (attempt {}) ===\n"
                "Your action '{}' had NO EFFECT because you are NOT adjacent to "
                "a crafting table. STOP trying to craft here. You must:\n"
                "  1. Move around to find your previously placed table, OR\n"
                "  2. Collect wood and place a NEW table (face grass, Place Table)\n"
                "Log this failure in actions_log.txt."
                .format(self._consecutive_craft_failures, prev_validated_action)
            )
        return (
            "=== CRAFTING FAILED ===\n"
            "Your action '{}' had NO EFFECT. This usually means you are NOT "
            "adjacent to a crafting table. Make sure you are near a placed table "
            "before trying to craft. Log this failure."
            .format(prev_validated_action)
        )

    def _get_craft_context(self, obs_text, prev_validated_action, had_effect):
        if self._env_name != "crafter":
            return ""

        parts = []

        if (prev_validated_action is not None and had_effect and
                prev_validated_action in _CRAFTER_PLACE_ACTIONS):
            if prev_validated_action == "Place Table":
                parts.append(
                    "=== YOU JUST PLACED A TABLE ===\n"
                    "CRAFT YOUR TOOLS NOW before moving away! Use 'Make Wood Pickaxe' "
                    "and 'Make Wood Sword' while you are still adjacent to the table. "
                    "If you walk away, you may not find the table again."
                )
            elif prev_validated_action == "Place Furnace":
                parts.append(
                    "=== YOU JUST PLACED A FURNACE ===\n"
                    "If you have the materials (wood + coal + iron) and are near a "
                    "table too, craft iron tools NOW before moving away."
                )

        obs_lower = obs_text.lower()
        if "table" in obs_lower and not parts:
            
            inv = self._parse_crafter_inventory_items(obs_text)
            has_wood = inv.get("wood", 0) > 0
            has_stone = inv.get("stone", 0) > 0
            has_pickaxe = (inv.get("wood_pickaxe", 0) > 0 or
                           inv.get("stone_pickaxe", 0) > 0 or
                           inv.get("iron_pickaxe", 0) > 0)
            has_sword = (inv.get("wood_sword", 0) > 0 or
                         inv.get("stone_sword", 0) > 0 or
                         inv.get("iron_sword", 0) > 0)

            hints = []
            if has_wood and not has_pickaxe:
                hints.append("Make Wood Pickaxe")
            if has_wood and not has_sword:
                hints.append("Make Wood Sword")
            if has_wood and has_stone and not inv.get("stone_pickaxe", 0) and not inv.get("iron_pickaxe", 0):
                hints.append("Make Stone Pickaxe")
            if has_wood and has_stone and not inv.get("stone_sword", 0) and not inv.get("iron_sword", 0):
                hints.append("Make Stone Sword")

            if hints:
                parts.append(
                    "=== TABLE NEARBY ===\n"
                    "A crafting table is visible. If you move adjacent to it, "
                    "you can craft: {}".format(", ".join(hints))
                )

        return "\n\n".join(parts)

    def _parse_crafter_visible_entities(self, obs_text):
        entities = []
        for line in obs_text.splitlines():
            line = line.strip()
            if line.startswith("- "):
                m = re.match(r"-\s+(\S+)\s+(\d+)\s+steps?\s+to\s+your\s+(.+)", line)
                if m:
                    entities.append((m.group(1).lower(), int(m.group(2)),
                                     m.group(3).strip()))
        
        m = re.search(r"You face (\S+) at your front", obs_text)
        if m:
            entities.append((m.group(1).lower(), 0, "in front"))
        return entities

    def _get_threat_warning(self, obs_text):
        if self._env_name != "crafter":
            return ""

        warnings = []
        vitals = self._parse_crafter_vitals(obs_text)
        current_health = vitals.get('health', 9)

        if self._prev_health is not None and current_health < self._prev_health:
            damage = self._prev_health - current_health
            warnings.append(
                "YOU TOOK {} DAMAGE last step! Health now {}/9. A hostile creature is attacking you.".format(damage, current_health)
            )

        visible = self._parse_crafter_visible_entities(obs_text)
        for entity, dist, direction in visible:
            if entity in _CRAFTER_HOSTILE_MOBS:
                if dist <= 2:
                    if current_health <= 3:
                        warnings.append(
                            "CRITICAL: {} {} step(s) {} — FLEE NOW! Health {}/9, another hit could kill you.".format(
                                entity, dist, direction, current_health)
                        )
                    else:
                        warnings.append(
                            "THREAT: {} {} step(s) {}. Fight (face it + Do) or move away to safety.".format(entity, dist, direction)
                        )
                elif dist <= 4 and entity == "skeleton":
                    warnings.append(
                        "CAUTION: skeleton {} steps {} — skeletons shoot arrows from range. Stay alert.".format(dist, direction)
                    )

        for entity, dist, direction in visible:
            if entity == "arrow" and dist <= 2:
                warnings.append(
                    "INCOMING ARROW {} step(s) {} — a skeleton is shooting at you!".format(dist, direction)
                )

        if not warnings:
            return ""
        return "=== COMBAT WARNING ===\n" + "\n".join(warnings)

    def _check_action_prerequisites(self, action, obs_text):
        if self._env_name != "crafter":
            return True, ""

        reqs = _CRAFTER_MATERIAL_REQUIREMENTS.get(action)
        if reqs is None:
            return True, ""

        inv = self._parse_crafter_inventory_items(obs_text)
        missing = []
        for item, needed in reqs.items():
            have = inv.get(item, 0)
            if have < needed:
                missing.append("need {} {}, you have {}".format(needed, item, have))

        if missing:
            return False, (
                "'{}' WILL FAIL — insufficient materials: {}. Collect the missing materials first, then try again."
                .format(action, "; ".join(missing))
            )
        return True, ""

    def _check_sleep_safety(self, action, obs_text):
        if action != "Sleep" or self._env_name != "crafter":
            return True, ""

        visible = self._parse_crafter_visible_entities(obs_text)
        for entity, dist, direction in visible:
            if entity in _CRAFTER_HOSTILE_MOBS and dist <= 3:
                return False, (
                    "Sleep is DANGEROUS — {} is {} step(s) {}! It will attack you while you sleep and you CANNOT defend. Move away from hostiles first, then sleep."
                    .format(entity, dist, direction)
                )
        return True, ""

    def _get_noop_warning(self):
        if self._env_name != "crafter":
            return ""
        if self._consecutive_noops >= 8:
            return (
                "=== NOOP SPIRAL ALERT ({} consecutive Noops) ===\n"
                "You have been doing NOTHING for {} steps in a row. "
                "This is wasting your entire game. You MUST take a productive action NOW. "
                "Check the TECH TREE STATUS and pursue the next goal."
                .format(self._consecutive_noops, self._consecutive_noops)
            )
        elif self._consecutive_noops >= 4:
            return (
                "=== NOOP WARNING ({} consecutive Noops) ===\n"
                "You have done nothing for {} steps. Take a productive action — "
                "explore, collect resources, or craft."
                .format(self._consecutive_noops, self._consecutive_noops)
            )
        return ""

    def _detect_inventory_progress(self, obs_text):
        if self._env_name != "crafter":
            return True
        current = self._parse_crafter_inventory_items(obs_text)
        if not self._prev_inventory_items:
            return True
        for key, val in current.items():
            old_val = self._prev_inventory_items.get(key, 0)
            if val > old_val:
                return True
        return False

    def _get_stagnation_warning(self):
        if self._env_name != "crafter":
            return ""
        s = self._steps_since_progress
        if s >= 50:
            return (
                "=== CRITICAL STAGNATION ({} steps, no new items) ===\n"
                "You have NOT gained any items in {} steps. Your current "
                "approach is FAILING. You MUST change strategy:\n"
                "  1. Check the TECH TREE STATUS for what to do next\n"
                "  2. If you need resources, explore in a NEW direction\n"
                "  3. If you need a table, collect wood (need 2) and place one"
                .format(s, s)
            )
        elif s >= 30:
            return (
                "=== STAGNATION WARNING ({} steps, no new items) ===\n"
                "No inventory gains in {} steps. Focus on the tech tree — "
                "collect resources or craft tools."
                .format(s, s)
            )
        return ""

    def _get_tech_tree_status(self, obs_text):
        if self._env_name != "crafter":
            return ""

        inv = self._parse_crafter_inventory_items(obs_text)
        visible = self._parse_crafter_visible_entities(obs_text)
        obs_lower = obs_text.lower()

        has_ipick = inv.get("iron_pickaxe", 0) > 0
        has_spick = inv.get("stone_pickaxe", 0) > 0
        has_wpick = inv.get("wood_pickaxe", 0) > 0
        has_isword = inv.get("iron_sword", 0) > 0
        has_ssword = inv.get("stone_sword", 0) > 0
        has_wsword = inv.get("wood_sword", 0) > 0

        best_pick = ("Iron" if has_ipick else "Stone" if has_spick
                     else "Wood" if has_wpick else None)
        best_sword = ("Iron" if has_isword else "Stone" if has_ssword
                      else "Wood" if has_wsword else None)

        wood = inv.get("wood", 0)
        stone = inv.get("stone", 0)
        coal = inv.get("coal", 0)
        iron = inv.get("iron", 0)
        sapling = inv.get("sapling", 0)

        table_nearby = "table" in obs_lower
        furnace_nearby = "furnace" in obs_lower

        nearest = {}
        for entity, dist, direction in visible:
            if entity not in nearest or dist < nearest[entity][0]:
                nearest[entity] = (dist, direction)

        lines = []

        tool_parts = []
        if best_pick:
            tool_parts.append(best_pick + " Pickaxe")
        if best_sword:
            tool_parts.append(best_sword + " Sword")
        lines.append("TOOLS: " + (", ".join(tool_parts) if tool_parts else "None"))

        mats = []
        if wood:
            mats.append("wood:{}".format(wood))
        if stone:
            mats.append("stone:{}".format(stone))
        if coal:
            mats.append("coal:{}".format(coal))
        if iron:
            mats.append("iron:{}".format(iron))
        if sapling:
            mats.append("sapling:{}".format(sapling))
        if mats:
            lines.append("MATERIALS: " + ", ".join(mats))

        next_line = None
        hint_line = None

        if best_pick is None:
            
            if wood >= 3 and table_nearby:
                next_line = (">>> Make Wood Pickaxe NOW (table nearby, have {} wood) <<<".format(wood))
            elif wood >= 4:
                next_line = ("Place Table (face GRASS tile, costs 2 wood). Then Make Wood Pickaxe + Wood Sword!")
            elif wood >= 3:
                next_line = ("Place Table (costs 2 wood, leaving 1 for pickaxe). Collect 1 more wood first for sword too!")
            else:
                need = max(0, 4 - wood)
                next_line = "Collect wood (Do on tree) — need {} more".format(need)
                if "tree" in nearest:
                    d, dr = nearest["tree"]
                    if d == 0:
                        hint_line = "Tree directly in front — use Do NOW"
                    else:
                        hint_line = "Nearest tree: {} step(s) {}".format(d, dr)
                else:
                    hint_line = "No trees visible — explore to find some"

        elif best_sword is None:
            
            if wood >= 1 and table_nearby:
                next_line = ">>> Make Wood Sword NOW (table nearby, have {} wood) <<<".format(wood)
            elif wood >= 3:
                next_line = "Place Table (costs 2 wood) then Make Wood Sword (costs 1 wood)"
            elif wood >= 1:
                
                next_line = "Find a table OR collect {} more wood to place new table + craft sword".format(3 - wood)
                if "tree" in nearest:
                    d, dr = nearest["tree"]
                    hint_line = "Nearest tree: {} step(s) {}".format(d, dr)
            else:
                next_line = "Collect wood (need 3: 2 for table + 1 for sword)"
                if "tree" in nearest:
                    d, dr = nearest["tree"]
                    hint_line = "Nearest tree: {} step(s) {}".format(d, dr)

        elif best_pick == "Wood":
            
            if stone >= 1 and wood >= 1 and table_nearby:
                next_line = ">>> Make Stone Pickaxe NOW — table nearby, have {} wood + {} stone <<<".format(wood, stone)
                if stone >= 2 and wood >= 2 and not has_ssword:
                    next_line += " Then Make Stone Sword!"
            elif stone == 0:
                next_line = "Mine stone (face stone, Do — Wood Pickaxe OK)"
                if "stone" in nearest:
                    d, dr = nearest["stone"]
                    if d == 0:
                        hint_line = "Stone directly in front — use Do NOW"
                    else:
                        hint_line = "Nearest stone: {} step(s) {}".format(d, dr)
                else:
                    hint_line = "No stone visible — explore to find some"
            elif not table_nearby:
                
                if wood >= 2:
                    next_line = "Place Table (face grass, costs 2 wood) to craft Stone Pickaxe"
                    if "grass" in nearest:
                        d, dr = nearest["grass"]
                        if d == 0:
                            hint_line = "Grass in front — Place Table now!"
                elif wood >= 1:
                    next_line = "Need 1 more wood to place table (costs 2), then craft Stone Pickaxe"
                    if "tree" in nearest:
                        d, dr = nearest["tree"]
                        hint_line = "Nearest tree: {} step(s) {}".format(d, dr)
                else:
                    next_line = "Collect 2 wood for table, then craft Stone Pickaxe (total need: 3 wood + 1 stone)"
                    if "tree" in nearest:
                        d, dr = nearest["tree"]
                        hint_line = "Nearest tree: {} step(s) {}".format(d, dr)
            elif wood == 0:
                next_line = "Collect wood for crafting (need 1+ for Stone Pickaxe)"
                if "tree" in nearest:
                    d, dr = nearest["tree"]
                    hint_line = "Nearest tree: {} step(s) {}".format(d, dr)

        elif best_pick == "Stone":
            
            if (iron >= 1 and coal >= 1 and wood >= 1 and
                    table_nearby and furnace_nearby):
                next_line = (">>> Make Iron Pickaxe NOW — table+furnace nearby, have {} wood + {} coal + {} iron <<<".format(wood, coal, iron))
                if (iron >= 2 and coal >= 2 and wood >= 2
                        and not has_isword):
                    next_line += " Then Make Iron Sword!"
            elif not furnace_nearby and stone >= 4:
                next_line = "Place Furnace (face GRASS, costs 4 stone — you have {})".format(stone)
                if table_nearby:
                    next_line += " — place it NEAR the table!"
            elif not furnace_nearby and stone < 4:
                
                need_stone = 4 - stone
                next_line = "Mine {} more stone for furnace (costs 4 stone, have {})".format(need_stone, stone)
                if "stone" in nearest:
                    d, dr = nearest["stone"]
                    if d == 0:
                        hint_line = "Stone directly in front — use Do NOW"
                    else:
                        hint_line = "Nearest stone: {} step(s) {}".format(d, dr)
                else:
                    hint_line = "No stone visible — explore to find some"
            elif iron == 0:
                next_line = "Mine iron (face iron, Do — Stone Pickaxe OK)"
                if "iron" in nearest:
                    d, dr = nearest["iron"]
                    if d == 0:
                        hint_line = "Iron directly in front — use Do NOW"
                    else:
                        hint_line = "Nearest iron: {} step(s) {}".format(d, dr)
                else:
                    hint_line = "No iron visible — explore to find some"
            elif coal == 0:
                next_line = "Mine coal (face coal, Do — pickaxe OK)"
                if "coal" in nearest:
                    d, dr = nearest["coal"]
                    if d == 0:
                        hint_line = "Coal directly in front — use Do NOW"
                    else:
                        hint_line = "Nearest coal: {} step(s) {}".format(d, dr)
                else:
                    hint_line = "No coal visible — explore to find some"
            elif not table_nearby:
                if wood >= 2:
                    next_line = "Place Table (face grass, costs 2 wood) near furnace for iron crafting"
                else:
                    next_line = "Collect wood (need 2 for table) for iron crafting setup"
                    if "tree" in nearest:
                        d, dr = nearest["tree"]
                        hint_line = "Nearest tree: {} step(s) {}".format(d, dr)

        elif best_pick == "Iron":
            
            next_line = "Mine diamond (face diamond, Do — Iron Pickaxe OK)"
            if "diamond" in nearest:
                d, dr = nearest["diamond"]
                if d == 0:
                    hint_line = "Diamond directly in front — use Do NOW"
                else:
                    hint_line = "Nearest diamond: {} step(s) {}".format(d, dr)
            else:
                hint_line = "No diamond visible — explore to find some"

        if next_line:
            lines.append("NEXT: " + next_line)
        if hint_line:
            lines.append("  -> " + hint_line)

        if best_pick == "Stone" and best_sword == "Wood":
            if stone >= 1 and wood >= 1 and table_nearby:
                lines.append(
                    "ALSO: Make Stone Sword (have materials, table nearby)")
        elif best_pick == "Iron" and best_sword != "Iron":
            if (iron >= 1 and coal >= 1 and wood >= 1
                    and table_nearby and furnace_nearby):
                lines.append(
                    "ALSO: Make Iron Sword (have materials, table+furnace nearby)")

        easy_wins = []
        if "Sleep" not in self._completed_actions and best_pick is not None:
            easy_wins.append("Use 'Sleep' to unlock Wake Up (free achievement)")
        if ("Place Stone" not in self._completed_actions
                and stone > 0 and best_pick is not None):
            easy_wins.append("Place Stone (face grass) — free achievement")
        if ("Place Plant" not in self._completed_actions
                and sapling > 0):
            easy_wins.append("Place Plant (face grass) — grows food")

        vitals = self._parse_crafter_vitals(obs_text)
        drink = vitals.get('drink', 9)
        if drink < 7 and "water" in nearest:
            d, dr = nearest["water"]
            easy_wins.append(
                "Collect Drink: water {} step(s) {}".format(d, dr))

        if easy_wins and len(lines) < 6:
            lines.append("EASY WINS: " + "; ".join(easy_wins[:2]))

        return "=== TECH TREE STATUS ===\n" + "\n".join(lines)

    def configure_memory(self, env_name, task, episode_idx=0):
        self._env_name = env_name
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
        self._recent_actions = []
        self._prev_inventory_items = {}
        self._consecutive_craft_failures = 0
        self._prev_health = None
        self._completed_actions = set()
        
        self._consecutive_noops = 0
        self._steps_since_progress = 0
        self._known_inventory_types = set()
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

        diff_notice = ""
        had_effect = True  
        if self._prev_obs_text is not None:
            had_effect = obs_text.strip() != self._prev_obs_text.strip()
            if not had_effect:
                self._unchanged_count += 1
                if self._unchanged_count >= 3:
                    diff_notice = ("CRITICAL WARNING: The observation has been UNCHANGED for {} consecutive actions. You are STUCK. Your recent actions have had NO EFFECT on the environment. You MUST try a COMPLETELY DIFFERENT approach -- do not repeat any of your recent actions.").format(self._unchanged_count)
                else:
                    diff_notice = ("WARNING: The observation is IDENTICAL to the previous step. Your previous action likely had NO EFFECT. This is important to log -- the action may be invalid or inapplicable in the current state.")
            else:
                self._unchanged_count = 0

        if self._prev_obs_text is not None and prev_validated_action is not None:
            self._recent_actions.append((prev_validated_action, had_effect))
            
            if len(self._recent_actions) > 20:
                self._recent_actions = self._recent_actions[-20:]

        if prev_validated_action == "Noop":
            self._consecutive_noops += 1
        elif prev_validated_action is not None:
            self._consecutive_noops = 0

        if self._detect_inventory_progress(obs_text):
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

        inventory_changes = self._get_inventory_changes(obs_text)

        craft_failure_notice = self._get_craft_failure_notice(
            prev_validated_action, had_effect)

        threat_warning = self._get_threat_warning(obs_text)

        tech_tree_status = self._get_tech_tree_status(obs_text)

        stagnation_warning = self._get_stagnation_warning()

        noop_warning = self._get_noop_warning()

        if prev_validated_action is not None and had_effect:
            self._completed_actions.add(prev_validated_action)

        map_analysis = ""
        if self._env_name in ("nle", "minihack"):
            map_analysis = parse_nle_map(obs_text)

        total_in = 0
        total_out = 0

        log_debug = None
        if prev_action is not None and self.step_count > 0:
            log_debug = self._log_phase(
                obs_text, prev_action, prev_validated_action,
                diff_notice, spatial_guidance, map_analysis,
                inventory_changes, craft_failure_notice,
                threat_warning,
            )
            total_in += log_debug["input_tokens"]
            total_out += log_debug["output_tokens"]

        plan_debug = self._plan_validate_phase(
            obs_text, instruction_prompt, diff_notice,
            spatial_guidance, map_analysis,
            prev_validated_action=prev_validated_action,
            had_effect=had_effect,
            tech_tree_status=tech_tree_status,
            threat_warning=threat_warning,
            stagnation_warning=stagnation_warning,
            noop_warning=noop_warning,
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

        if self._env_name == "crafter":
            crafter_repaired, crafter_was_repaired = repair_crafter_action(final_action)
            if crafter_was_repaired:
                logger.info(
                    "Crafter action repaired: '%s' -> '%s'",
                    final_action, crafter_repaired,
                )
                final_action = crafter_repaired

        self.last_step_debug = {
            "step": self.step_count,
            "log_phase": log_debug,
            "plan_phase": plan_debug,
            "memory_snapshot": self.memory.snapshot(),
            "diff_notice": diff_notice,
            "unchanged_count": self._unchanged_count,
            "map_analysis": map_analysis,
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
        self._prev_inventory_items = self._parse_crafter_inventory_items(obs_text)
        
        vitals = self._parse_crafter_vitals(obs_text)
        self._prev_health = vitals.get('health', 9)
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
                   diff_notice, spatial_guidance, map_analysis,
                   inventory_changes="", craft_failure_notice="",
                   threat_warning=""):
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

        vital_urgency = self._get_vital_urgency(obs_text)
        compaction_warnings = self._get_compaction_warnings()
        env_specific_logging = self._get_env_specific_logging()

        initial_prompt = LOG_SYSTEM.format(
            action_description=action_description,
            diff_notice=diff_notice,
            inventory_changes=inventory_changes,
            craft_failure_notice=craft_failure_notice,
            threat_warning=threat_warning,
            vital_urgency=vital_urgency,
            spatial_guidance=spatial_guidance,
            map_analysis=map_analysis,
            observation=obs_text,
            file_index=file_index,
            compaction_warnings=compaction_warnings,
            env_specific_logging=env_specific_logging,
        )

        messages = [Message(role="user", content=initial_prompt)]

        total_in = 0
        total_out = 0
        turns = []

        for iteration in range(self.max_log_iterations):
            current_prompt = messages[-1].content

            try:
                response = self.client.generate(messages)
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

        return {
            "initial_prompt": initial_prompt,
            "turns": turns,
            "input_tokens": total_in,
            "output_tokens": total_out,
        }

    def _plan_validate_phase(self, obs_text, instruction_prompt,
                             diff_notice, spatial_guidance, map_analysis,
                             prev_validated_action=None, had_effect=True,
                             tech_tree_status="", threat_warning="",
                             stagnation_warning="", noop_warning=""):
        file_index = self.memory.file_index()

        stuckness_warning = ""
        if diff_notice:
            stuckness_warning = diff_notice
            if prev_validated_action:
                stuckness_warning += (
                    "\nYour last executed action was '{}'. "
                    "Do NOT repeat it -- choose something different."
                ).format(prev_validated_action)

        vital_urgency = self._get_vital_urgency(obs_text)
        repetition_warning = self._get_repetition_warning()
        goals_content = self._get_goals_content()
        env_specific_planning = self._get_env_specific_planning()

        craft_context = self._get_craft_context(
            obs_text, prev_validated_action, had_effect)

        initial_prompt = PLAN_SYSTEM.format(
            instruction_prompt=instruction_prompt,
            spatial_guidance=spatial_guidance,
            stuckness_warning=stuckness_warning,
            vital_urgency=vital_urgency,
            threat_warning=threat_warning,
            repetition_warning=repetition_warning,
            craft_context=craft_context,
            tech_tree_status=tech_tree_status,
            stagnation_warning=stagnation_warning,
            noop_warning=noop_warning,
            map_analysis=map_analysis,
            goals_content=goals_content,
            file_index=file_index,
            observation=obs_text,
            env_specific_planning=env_specific_planning,
        )

        messages = [Message(role="user", content=initial_prompt)]

        total_in = 0
        total_out = 0
        turns = []
        final_action = None
        model_id = "unknown"
        
        prereq_rejections = 0

        for iteration in range(self.max_validate_iterations):
            current_prompt = messages[-1].content

            try:
                response = self.client.generate(messages)
            except Exception as e:
                logger.error("Plan phase LLM call failed: %s", e)
                turns.append({
                    "prompt": current_prompt,
                    "raw_response": "[ERROR: {}]".format(e),
                    "tool_results": "",
                    "committed_action": None,
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
            })

            messages.append(Message(role="assistant", content=assistant_text))

            if committed_action is not None:
                
                if self._env_name == "crafter" and prereq_rejections < 2:
                    
                    mat_valid, mat_msg = self._check_action_prerequisites(
                        committed_action, obs_text)
                    
                    sleep_safe, sleep_msg = self._check_sleep_safety(
                        committed_action, obs_text)
                    
                    noop_ok = True
                    noop_msg = ""
                    if (committed_action == "Noop"
                            and self._consecutive_noops >= 5):
                        noop_ok = False
                        noop_msg = (
                            "Noop REJECTED — you have already done Noop {} consecutive times. Choose a PRODUCTIVE action."
                            .format(self._consecutive_noops)
                        )

                    if not mat_valid or not sleep_safe or not noop_ok:
                        prereq_rejections += 1
                        reason = mat_msg or sleep_msg or noop_msg
                        reject_msg = (
                            "ACTION REJECTED: {}\n"
                            "Choose a DIFFERENT action and commit with "
                            "<|ACTION|>your_action<|END|>."
                        ).format(reason)
                        messages.append(Message(
                            role="user", content=reject_msg))
                        logger.info(
                            "prereq gate rejected '%s': %s",
                            committed_action, reason,
                        )
                        continue  

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
