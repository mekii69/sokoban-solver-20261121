// Sokoban solver: push-level search, solved room by room.
//
// Idea: a level is a hallway plus small rooms (3-4 boxes each). A box pushed
// out of its room can never come back, so each room can be solved on its own
// with the other rooms' boxes treated as walls, and the room optima add up to
// the level optimum. Inside a room we run a breadth-first search over pushes
// (first solved state found is push-optimal), with two safe prunings: squares
// from which a lone box can never reach a goal, and frozen 2x2 blocks. The
// player's walking moves between pushes are filled in with shortest paths.
// Work-based budgets only: no clock, no randomness, no network, so two runs
// produce identical bytes.
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Four steps, in a fixed order so every tie breaks the same way.
const STEPS = [
  { dx: 0, dy: -1, walk: 'u', push: 'U' },
  { dx: 0, dy: 1, walk: 'd', push: 'D' },
  { dx: -1, dy: 0, walk: 'l', push: 'L' },
  { dx: 1, dy: 0, walk: 'r', push: 'R' },
];

const ROOM_STATE_CAP = 350000;
const WHOLE_STATE_CAP = 500000;
const GREEDY_STATE_CAP = 500000;
const TOTAL_STATE_BUDGET = 2500000;

let spentStates = 0;

interface Board {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

const at = (board: Board, i: number, dx: number, dy: number): number => {
  const x = (i % board.w) + dx;
  const y = Math.floor(i / board.w) + dy;
  if (x < 0 || y < 0 || x >= board.w || y >= board.h) return -1;
  return y * board.w + x;
};

// Squares a single box can be pushed from and still reach some goal:
// start at the goals and pull boxes backwards (a box on `c` pushed along
// `d` came from `c - d`, with the player standing at `c - 2d`).
function liveCells(board: Board): Uint8Array {
  const live = new Uint8Array(board.w * board.h);
  const queue: number[] = [];
  for (const g of board.goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi];
    for (const s of STEPS) {
      const from = at(board, c, -s.dx, -s.dy);
      const pusher = at(board, c, -2 * s.dx, -2 * s.dy);
      if (from < 0 || pusher < 0) continue;
      if (board.wall[from] || board.wall[pusher] || live[from]) continue;
      live[from] = 1;
      queue.push(from);
    }
  }
  return live;
}

// Flood fill from `start` avoiding walls and `blocked` cells. Marks `seen`
// with `stamp`, returns the smallest cell reached (the region's identity).
function flood(
  board: Board,
  blocked: Uint8Array,
  start: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = start;
  let n = 0;
  stack[n++] = start;
  seen[start] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (const s of STEPS) {
      const q = at(board, p, s.dx, s.dy);
      if (q < 0 || board.wall[q] || blocked[q] || seen[q] === stamp) continue;
      seen[q] = stamp;
      stack[n++] = q;
    }
  }
  return low;
}

// True when the box now on `t` is stuck inside a 2x2 block of walls/boxes
// holding at least one box off a goal: no push can ever free such a block.
function frozenBlock(
  board: Board,
  blocked: Uint8Array,
  isGoal: Uint8Array,
  t: number,
): boolean {
  const w = board.w;
  const corners = [t, t - 1, t - w, t - w - 1];
  for (const a of corners) {
    const ax = a % w;
    const ay = Math.floor(a / w);
    if (ax < 0 || ay < 0 || ax + 1 >= w || ay + 1 >= board.h) continue;
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    const solid = (i: number): boolean => board.wall[i] === 1 || blocked[i] === 1;
    if (!solid(a) || !solid(b) || !solid(c) || !solid(d)) continue;
    if ((blocked[a] && !isGoal[a]) || (blocked[b] && !isGoal[b]) ||
        (blocked[c] && !isGoal[c]) || (blocked[d] && !isGoal[d])) return true;
  }
  return false;
}

// Shortest walk (lowercase letters) from `from` to `to` around walls/boxes.
function walkPath(
  board: Board,
  blocked: Uint8Array,
  from: number,
  to: number,
): string {
  if (from === to) return '';
  const prev = new Int32Array(board.w * board.h).fill(-1);
  const how: string[] = new Array(board.w * board.h);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const s of STEPS) {
      const q = at(board, p, s.dx, s.dy);
      if (q < 0 || board.wall[q] || blocked[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = s.walk;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('push square unreachable');
  let out = '';
  for (let p = to; p !== from; p = prev[p]) out = how[p] + out;
  return out;
}

function applyMoves(board: Board, player: number, moves: string): number {
  let p = player;
  for (const ch of moves) {
    const s = STEPS.find((x) => x.walk === ch.toLowerCase())!;
    p = at(board, p, s.dx, s.dy);
  }
  return p;
}

interface SearchResult {
  pushes: number;
  moves: string;
}

// Push-optimal BFS over one board (a room, or the whole level as fallback).
// `numericKey` is safe only when boxes^count fits exactly (rooms: <=4 boxes).
function searchOptimal(
  board: Board,
  maxStates: number,
  numericKey: boolean,
): SearchResult | null {
  const size = board.w * board.h;
  const isGoal = new Uint8Array(size);
  for (const g of board.goals) isGoal[g] = 1;
  const live = liveCells(board);
  if (board.boxes.some((b) => !live[b])) return null;

  const blocked = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const keyOf = (boxes: number[], low: number): number | string => {
    if (!numericKey) return low + '|' + boxes.join(',');
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };

  const allSolved = (boxes: number[]): boolean => boxes.every((b) => isGoal[b]);

  const boxesList: number[][] = [board.boxes.slice()];
  const playerList: number[] = [board.player];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const visited = new Map<number | string, number>();
  for (const b of board.boxes) blocked[b] = 1;
  const startLow = flood(board, blocked, board.player, seen, ++stamp, stack);
  for (const b of board.boxes) blocked[b] = 0;
  visited.set(keyOf(board.boxes, startLow), 0);

  let found = allSolved(board.boxes) ? 0 : -1;
  for (let head = 0; found < 0 && head < boxesList.length; head++) {
    const boxes = boxesList[head];
    for (const b of boxes) blocked[b] = 1;
    flood(board, blocked, playerList[head], seen, ++stamp, stack);
    const regionStamp = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      for (const s of STEPS) {
        const dest = at(board, b, s.dx, s.dy);
        const behind = at(board, b, -s.dx, -s.dy);
        if (dest < 0 || behind < 0) continue;
        if (seen[behind] !== regionStamp) continue; // player cannot reach the push square
        if (board.wall[dest] || blocked[dest] || !live[dest]) continue;
        blocked[b] = 0;
        blocked[dest] = 1;
        const frozen = frozenBlock(board, blocked, isGoal, dest);
        let key: number | string = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = dest;
          next.sort((a, c) => a - c);
          const low = flood(board, blocked, b, mark, ++stamp, stack);
          key = keyOf(next, low);
        }
        blocked[dest] = 0;
        blocked[b] = 1;
        if (frozen || visited.has(key)) continue;
        if (boxesList.length >= maxStates || spentStates >= TOTAL_STATE_BUDGET) {
          for (const x of boxes) blocked[x] = 0;
          return null;
        }
        spentStates++;
        visited.set(key, boxesList.length);
        boxesList.push(next!);
        playerList.push(b);
        parent.push(head);
        pushFrom.push(b);
        pushTo.push(dest);
        if (allSolved(next!)) {
          found = boxesList.length - 1;
          break;
        }
      }
    }
    for (const x of boxes) blocked[x] = 0;
  }
  if (found < 0) return null;

  // Rebuild: follow parents back, then replay pushes forward with walks.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = board.player;
  let cur = board.boxes.slice();
  for (const s of path) {
    const from = pushFrom[s];
    const to = pushTo[s];
    const step = STEPS.find((x) => at(board, from, x.dx, x.dy) === to)!;
    for (const b of cur) blocked[b] = 1;
    moves += walkPath(board, blocked, player, at(board, from, -step.dx, -step.dy));
    for (const b of cur) blocked[b] = 0;
    moves += step.push;
    player = from;
    cur = boxesList[s].slice();
  }
  return { pushes: path.length, moves };
}

interface Room {
  boxes: number[];
  goals: number[];
  done: boolean;
}

// Split the board into rooms: cut the dead squares of the player's row
// (the hallway: a box there reaches no goal) and take each remaining
// connected part holding boxes as a room.
function splitRooms(board: Board): Room[] | null {
  const live = liveCells(board);
  const row = Math.floor(board.player / board.w);
  const cut = new Uint8Array(board.w * board.h);
  for (let i = row * board.w; i < (row + 1) * board.w; i++) {
    if (!board.wall[i] && !live[i]) cut[i] = 1;
  }
  if (board.boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(board.w * board.h).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < board.w * board.h; s++) {
    if (board.wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const st of STEPS) {
        const q = at(board, squares[i], st.dx, st.dy);
        if (q < 0 || board.wall[q] || cut[q] || part[q] >= 0) continue;
        part[q] = rooms.length;
        squares.push(q);
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of board.boxes) rooms[part[b]].boxes.push(b);
  for (const g of board.goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// Solve room by room; null when any room's optimal search gives up.
function solveByRooms(board: Board): SearchResult | null {
  const rooms = splitRooms(board);
  if (rooms === null) return null;
  let player = board.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = board.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.boxes) wall[b] = 1;
    }
    const sub: Board = {
      w: board.w,
      h: board.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = searchOptimal(sub, ROOM_STATE_CAP, true);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = applyMoves(board, player, solved.moves);
    // The room's boxes now sit on its goals; keep them as walls for later rooms.
    room.boxes = room.goals.slice();
    room.done = true;
  }
  return { pushes, moves };
}

// Greedy fallback: best-first push search (fewest pushes + nearest-goal
// estimate first) that returns any valid solution when optimal search fails.
function solveGreedy(board: Board): SearchResult | null {
  const size = board.w * board.h;
  const isGoal = new Uint8Array(size);
  for (const g of board.goals) isGoal[g] = 1;
  const live = liveCells(board);
  if (board.boxes.some((b) => !live[b])) return null;
  const blocked = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const estimate = (boxes: number[]): number => {
    let sum = 0;
    for (const b of boxes) {
      let best = Infinity;
      for (const g of board.goals) {
        const d = Math.abs((b % board.w) - (g % board.w)) +
          Math.abs(Math.floor(b / board.w) - Math.floor(g / board.w));
        if (d < best) best = d;
      }
      sum += best;
    }
    return sum;
  };
  const boxesList: number[][] = [board.boxes.slice()];
  const playerList: number[] = [board.player];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const pushesOf: number[] = [0];
  const bestSeen = new Map<string, number>();
  for (const b of board.boxes) blocked[b] = 1;
  const startLow = flood(board, blocked, board.player, seen, ++stamp, stack);
  for (const b of board.boxes) blocked[b] = 0;
  bestSeen.set(startLow + '|' + board.boxes.join(','), 0);
  // Binary heap of [priority, pushes, state index]; deterministic ties.
  const heap: [number, number, number][] = [[estimate(board.boxes), 0, 0]];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const heapPop = (): [number, number, number] => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };
  while (heap.length > 0) {
    const [, g, head] = heapPop();
    if (g !== pushesOf[head]) continue;
    const boxes = boxesList[head];
    if (boxes.every((b) => isGoal[b])) {
      const path: number[] = [];
      for (let s = head; s > 0; s = parent[s]) path.push(s);
      path.reverse();
      let moves = '';
      let player = board.player;
      let cur = board.boxes.slice();
      for (const s of path) {
        const from = pushFrom[s];
        const to = pushTo[s];
        const step = STEPS.find((x) => at(board, from, x.dx, x.dy) === to)!;
        for (const b of cur) blocked[b] = 1;
        moves += walkPath(board, blocked, player, at(board, from, -step.dx, -step.dy));
        for (const b of cur) blocked[b] = 0;
        moves += step.push;
        player = from;
        cur = boxesList[s].slice();
      }
      return { pushes: path.length, moves };
    }
    for (const b of boxes) blocked[b] = 1;
    flood(board, blocked, playerList[head], seen, ++stamp, stack);
    const regionStamp = stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (const s of STEPS) {
        const dest = at(board, b, s.dx, s.dy);
        const behind = at(board, b, -s.dx, -s.dy);
        if (dest < 0 || behind < 0) continue;
        if (seen[behind] !== regionStamp) continue;
        if (board.wall[dest] || blocked[dest] || !live[dest]) continue;
        blocked[b] = 0;
        blocked[dest] = 1;
        const frozen = frozenBlock(board, blocked, isGoal, dest);
        let key = '';
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = dest;
          next.sort((a, c) => a - c);
          const low = flood(board, blocked, b, mark, ++stamp, stack);
          key = low + '|' + next.join(',');
        }
        blocked[dest] = 0;
        blocked[b] = 1;
        if (frozen) continue;
        const prev = bestSeen.get(key);
        if (prev !== undefined && prev <= g + 1) continue;
        if (boxesList.length >= GREEDY_STATE_CAP || spentStates >= TOTAL_STATE_BUDGET) {
          for (const x of boxes) blocked[x] = 0;
          return null;
        }
        spentStates++;
        bestSeen.set(key, g + 1);
        const idx = boxesList.length;
        boxesList.push(next!);
        playerList.push(b);
        parent.push(head);
        pushFrom.push(b);
        pushTo.push(dest);
        pushesOf.push(g + 1);
        heapPush([g + 1 + estimate(next!), g + 1, idx]);
      }
    }
    for (const x of boxes) blocked[x] = 0;
  }
  return null;
}

function solveLevel(text: string): string {
  const board = parseLevel(text);
  if (board.player < 0 || board.boxes.length === 0) return '';
  const byRooms = solveByRooms(board);
  if (byRooms !== null) return byRooms.moves;
  const whole = searchOptimal(board, WHOLE_STATE_CAP, false);
  if (whole !== null) return whole.moves;
  const greedy = solveGreedy(board);
  if (greedy !== null) return greedy.moves;
  return '';
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  // One level failing must never take the other four with it: on any
  // unexpected error, still write a well-formed (empty) answer for it.
  let moves = '';
  try {
    const levelFile = join(INPUT_DIR, id, 'level.txt');
    if (existsSync(levelFile)) moves = solveLevel(readFileSync(levelFile, 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
