# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Writes a synthetic but plausible cuju recording, a .cjr file, for testing the viewer.

The match has players jogging around a formation with a procedural gait in the 14 skeleton
nodes, a ball passed between them, goals, kick offs and other restarts, half time and full time.
Nothing here depends on cuju, so the viewer can be tested without building cuju.

Usage:
    python ci/render_sample.py out.cjr [--players 11] [--steps 1200] [--seed 2]
"""

import argparse
import gzip
import json
import math
import random
import struct

MAGIC = b'CUJREC1\n'
NODES = [
    'player',
    'body',
    'middle',
    'neck',
    'left_shoulder',
    'left_elbow',
    'right_shoulder',
    'right_elbow',
    'left_thigh',
    'left_knee',
    'left_ankle',
    'right_thigh',
    'right_knee',
    'right_ankle',
]
PARENTS = [-1, 0, 1, 2, 2, 4, 2, 6, 1, 8, 9, 1, 11, 12]

# Local rest positions from GRF's player.object. GRF's animations replace every local rotation, so
# the rest pose, the bind pose of the viewer's player.glb, has identity rotations.
REST = [
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.96),
    (0.0, 0.0, 0.15),
    (0.0, -0.03, 0.5),
    (0.16, -0.01, 0.48),
    (-0.01, 0.0, -0.33),
    (-0.16, -0.01, 0.48),
    (0.01, 0.0, -0.33),
    (0.087, 0.0, -0.01),
    (0.0, 0.0, -0.42),
    (0.0, -0.04, -0.44),
    (-0.087, 0.0, -0.01),
    (0.0, 0.0, -0.42),
    (0.0, -0.04, -0.44),
]

PITCH_HALF_W = 55.0
PITCH_HALF_H = 36.0
GOAL_HALF_WIDTH = 3.7
BALL_RADIUS = 0.11
# A ball that moves farther in one step was put down for a restart.
MAX_BALL_TRAVEL = 4.0
STEPS_PER_SECOND = 10
DT = 1.0 / STEPS_PER_SECOND
FULL_MATCH_MS = 90 * 60 * 1000

NORMAL, KICK_OFF, GOAL_KICK, FREE_KICK, CORNER, THROW_IN, PENALTY = range(7)

# GRF's e_FunctionType, the kind of animation a player runs, as the action bytes store it.
(
    NONE,
    MOVEMENT,
    BALL_CONTROL,
    TRAP,
    SHORT_PASS,
    LONG_PASS,
    HIGH_PASS,
    HEADER,
    SHOT,
    DEFLECT,
    CATCH,
    INTERFERE,
    TRIP,
    SLIDING,
    SPECIAL,
) = range(15)

# The touch bytes hold one more than GRF's e_TouchType, and 0 for no touch during the step.
NO_TOUCH, KICKED, NON_KICKED, ACCIDENTAL = range(4)

# GRF's kinds of foul, as the referee fields of a frame store them.
NO_FOUL, FOUL, YELLOW_CARD, RED_CARD = range(4)

# How many steps an action shows before the player falls back to movement.
ACTION_STEPS = 6

# Formation spots in the left team's frame, x towards the opponent goal, as fractions of the
# half pitch. Index 0 is the goalkeeper.
FORMATION_11 = [
    (-0.95, 0.0),
    (-0.65, -0.55),
    (-0.7, -0.2),
    (-0.7, 0.2),
    (-0.65, 0.55),
    (-0.3, -0.6),
    (-0.35, -0.2),
    (-0.35, 0.2),
    (-0.3, 0.6),
    (-0.05, -0.15),
    (-0.05, 0.15),
]
FORMATION_5 = [(-0.95, 0.0), (-0.6, -0.35), (-0.6, 0.35), (-0.25, -0.2), (-0.1, 0.2)]


def quat_axis_angle(axis, angle):
    """Returns the quaternion x, y, z, w of a rotation by angle radians about an axis.

    Args:
        axis: Rotation axis, three floats, need not be unit length.
        angle: Rotation angle in radians.

    Returns:
        The quaternion as a tuple x, y, z, w.
    """
    n = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    if n == 0.0 or angle == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    s = math.sin(angle * 0.5) / n
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle * 0.5))


def quat_mul(a, b):
    """Returns the Hamilton product a times b of two quaternions stored x, y, z, w."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_rotate(q, v):
    """Rotates the vector v by the unit quaternion q."""
    x, y, z, w = q
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + y * tz - z * ty,
        v[1] + w * ty + z * tx - x * tz,
        v[2] + w * tz + x * ty - y * tx,
    )


class Player:
    """One synthetic player, its position, heading and gait phase."""

    def __init__(self, team, index, spot, goalkeeper, rng):
        self.team = team
        self.index = index
        self.spot = spot
        self.goalkeeper = goalkeeper
        self.pos = list(spot)
        self.vel = [0.0, 0.0]
        self.heading = 0.0 if team == 0 else math.pi
        self.phase = rng.random() * math.tau
        self.kick = 0.0
        self.action = MOVEMENT
        self.action_steps = 0
        self.touch = NO_TOUCH

    def act(self, action, touch=NO_TOUCH):
        """Starts an action for a few steps and marks a touch of the ball on this step."""
        self.action = action
        self.action_steps = ACTION_STEPS
        if touch:
            self.touch = touch

    def skeleton(self):
        """Returns the 14 world transforms as lists of position xyz then rotation xyzw."""
        speed = math.hypot(self.vel[0], self.vel[1])
        stride = min(1.0, speed / 6.0)
        s = math.sin(self.phase)
        c = math.cos(self.phase)
        bob = 0.03 * stride * abs(c)
        lean = -0.12 * stride
        # Positive swing moves a limb forward, which is a rotation about local x by minus angle.
        thigh_l = 0.55 * stride * s
        thigh_r = -0.55 * stride * s
        knee_l = 0.15 + 0.9 * stride * max(0.0, -c)
        knee_r = 0.15 + 0.9 * stride * max(0.0, c)
        if self.kick > 0.0:
            swing = math.sin(self.kick * math.pi)
            thigh_r = 0.9 * swing
            knee_r = 0.2 + 0.6 * (1.0 - swing)
        arm_l = -0.5 * stride * s
        arm_r = 0.5 * stride * s
        anim = [
            (0.0, 0.0, 0.0),
            (lean, 0.0, 0.0),
            (lean * 0.5, 0.0, 0.0),
            (-lean, 0.0, 0.0),
            (-arm_l, 0.0, 0.0),
            (-0.5 - 0.4 * stride, 0.0, 0.0),
            (-arm_r, 0.0, 0.0),
            (-0.5 - 0.4 * stride, 0.0, 0.0),
            (-thigh_l, 0.0, 0.0),
            (knee_l, 0.0, 0.0),
            (-0.2 * stride * s, 0.0, 0.0),
            (-thigh_r, 0.0, 0.0),
            (knee_r, 0.0, 0.0),
            (0.2 * stride * s, 0.0, 0.0),
        ]
        # The model faces local minus y, so a heading h needs a yaw of h plus a quarter turn.
        root_rot = quat_axis_angle((0.0, 0.0, 1.0), self.heading + math.pi / 2)
        world = []
        for i, rest_pos in enumerate(REST):
            local_rot = quat_axis_angle((1.0, 0.0, 0.0), anim[i][0])
            if i == 0:
                pos = (self.pos[0], self.pos[1], 0.0)
                rot = root_rot
            else:
                ppos, prot = world[PARENTS[i]]
                offset = rest_pos
                if i == 1:
                    offset = (0.0, 0.0, rest_pos[2] - bob)
                d = quat_rotate(prot, offset)
                pos = (ppos[0] + d[0], ppos[1] + d[1], ppos[2] + d[2])
                rot = quat_mul(prot, local_rot)
            world.append((pos, rot))
        return [list(p) + list(r) for p, r in world]


class Match:
    """A small rough simulation, good enough to look like football."""

    def __init__(self, per_team, steps, seed, names, teams=None):
        self.rng = random.Random(seed)
        self.steps = steps
        self.per_team = per_team
        self.names = names
        self.teams = teams
        formation = FORMATION_11 if per_team == 11 else FORMATION_5
        self.players = []
        for team in (0, 1):
            sign = 1.0 if team == 0 else -1.0
            for i in range(per_team):
                fx, fy = formation[i % len(formation)]
                spot = (sign * fx * PITCH_HALF_W, sign * fy * PITCH_HALF_H)
                self.players.append(Player(team, i, spot, i == 0, self.rng))
        self.ball = [0.0, 0.0, BALL_RADIUS]
        self.ball_vel = [0.0, 0.0, 0.0]
        self.ball_rotation = (0.0, 0.0, 0.0, 1.0)
        self.ball_spin = (0.0, 0.0, 0.0, 1.0)
        self.score = [0, 0]
        self.mode = KICK_OFF
        self.mode_timer = 8
        self.owner = -1
        self.hold = 0
        self.kick_team = 0
        self.step_index = 0
        self.half_done = False
        # The referee's last decision, kept while play waits for the restart, and the first
        # player in an offside position at the last pass.
        self.foul = (-1, -1, NO_FOUL)
        self.offside = -1
        # Each kind of decision is also scheduled once at a fraction of the match, so even a
        # short sample shows a foul, a yellow card, an offside call and a red card.
        self.shown = set()
        self.scheduled = {
            round(steps * fraction): kind
            for fraction, kind in ((0.2, 'foul'), (0.35, 'yellow'), (0.55, 'offside'), (0.8, 'red'))
        }
        self.offside_at = round(steps * 0.55)
        # Each player's flags byte, 1 on the pitch, 2 a yellow card, 4 a red card.
        self.cards = [1] * len(self.players)

    def header(self):
        """Returns the recording header as a dict."""
        count = len(self.players)
        return {
            'version': 1,
            'title': f'{self.names[0]} against {self.names[1]}, synthetic sample',
            'scenario': '11_vs_11_stochastic' if self.per_team == 11 else '5_vs_5',
            'steps_per_second': STEPS_PER_SECOND,
            'nodes': NODES,
            'parents': PARENTS,
            'players': [
                {
                    'team': p.team,
                    'index': p.index,
                    'goalkeeper': p.goalkeeper,
                    'controlled': p.index == (self.per_team - 1) and p.team == 0,
                }
                for p in self.players
            ],
            'teams': self.teams
            or [
                {'name': self.names[0], 'color': [200, 30, 40], 'secondary': [250, 250, 250]},
                {'name': self.names[1], 'color': [30, 80, 200], 'secondary': [20, 20, 40]},
            ],
            'frame_bytes': frame_bytes(count),
            'frame_count': self.steps,
        }

    def reset_positions(self, kick_team):
        for p in self.players:
            sx, sy = p.spot
            p.pos = [min(sx, -1.0) if p.team == 0 else max(sx, 1.0), sy]
            p.vel = [0.0, 0.0]
        self.ball = [0.0, 0.0, BALL_RADIUS]
        self.ball_vel = [0.0, 0.0, 0.0]
        self.kick_team = kick_team
        self.owner = self.nearest(kick_team, self.ball)
        self.mode = KICK_OFF
        self.mode_timer = 10

    def nearest(self, team, point):
        best, best_d = -1, 1e9
        for i, p in enumerate(self.players):
            if p.team != team:
                continue
            d = math.hypot(p.pos[0] - point[0], p.pos[1] - point[1])
            if d < best_d:
                best, best_d = i, d
        return best

    def restart(self, mode, team, spot):
        self.mode = mode
        self.mode_timer = 12
        self.ball = [spot[0], spot[1], BALL_RADIUS]
        self.ball_vel = [0.0, 0.0, 0.0]
        self.owner = self.nearest(team, spot)
        self.kick_team = team
        owner = self.players[self.owner]
        owner.pos = [spot[0] - (1.0 if team == 0 else -1.0) * 0.6, spot[1]]

    def pass_ball(self):
        """The owner passes forward to a teammate, or shoots when close to goal."""
        owner = self.players[self.owner]
        attack = 1.0 if owner.team == 0 else -1.0
        goal_x = attack * PITCH_HALF_W
        distance_to_goal = abs(goal_x - owner.pos[0])
        roll = self.rng.random()
        if distance_to_goal < 36.0 and roll < 0.75:
            target = (goal_x + attack * 1.0, self.rng.uniform(-6.0, 6.0))
            speed = 26.0
            lift = self.rng.uniform(0.0, 4.0)
            action = SHOT
        elif roll > 0.85:
            # A clearance, often out of play, so the sample shows throw ins and goal kicks.
            target = (
                owner.pos[0] + attack * self.rng.uniform(20.0, 70.0),
                self.rng.choice((-1.0, 1.0)) * self.rng.uniform(20.0, 45.0),
            )
            speed = 25.0
            lift = self.rng.uniform(3.0, 8.0)
            action = LONG_PASS
        else:
            mates = [
                i
                for i, p in enumerate(self.players)
                if p.team == owner.team and i != self.owner and not p.goalkeeper
            ]
            weights = [
                max(0.2, 1.0 + attack * (self.players[i].pos[0] - owner.pos[0]) / 20.0)
                for i in mates
            ]
            mate = self.rng.choices(mates, weights)[0]
            target = tuple(self.players[mate].pos)
            self.offside = mate if self.beyond_defence(mate) else -1
            if self.offside >= 0 and self.call('offside', 0.35):
                # The referee stops play for a free kick to the defence where the attacker stood.
                defence = 1 - owner.team
                owner.act(SHORT_PASS, KICKED)
                self.restart(FREE_KICK, defence, target)
                return
            speed = self.rng.uniform(12.0, 20.0)
            lift = 6.0 if self.rng.random() < 0.15 else 0.0
            action = HIGH_PASS if lift else SHORT_PASS
        dx = target[0] - self.ball[0]
        dy = target[1] - self.ball[1]
        d = math.hypot(dx, dy) or 1.0
        self.ball_vel = [dx / d * speed, dy / d * speed, lift]
        owner.kick = 1e-3
        owner.act(action, KICKED)
        self.owner = -1
        self.hold = 0

    def call(self, kind, chance):
        """Returns whether the referee makes a call of kind, always the first time."""
        if kind not in self.shown:
            self.shown.add(kind)
            return True
        return self.rng.random() < chance

    def beyond_defence(self, mate):
        """Returns whether a teammate stands beyond the second last defender and the ball."""
        p = self.players[mate]
        attack = 1.0 if p.team == 0 else -1.0
        depths = sorted((attack * q.pos[0] for q in self.players if q.team != p.team), reverse=True)
        line = max(depths[1] if len(depths) > 1 else 0.0, attack * self.ball[0], 0.0)
        return attack * p.pos[0] > line + 0.5

    def scheduled_call(self):
        """Makes the decision scheduled for this step, once play runs with a player on the ball."""
        due = [step for step in self.scheduled if step <= self.step_index]
        if not due or self.mode != NORMAL or self.owner < 0:
            return False
        kind = self.scheduled.pop(min(due))
        if kind in self.shown and kind != 'offside':
            return False
        owner = self.players[self.owner]
        if kind == 'offside':
            striker = next(
                i
                for i, p in enumerate(self.players)
                if p.team == owner.team and p.index == self.per_team - 1
            )
            if striker == self.owner:
                # The striker has the ball himself, the call waits for a pass to him.
                self.scheduled[self.step_index + 1] = kind
                return False
            self.shown.add('offside')
            self.offside = striker
            owner.act(SHORT_PASS, KICKED)
            self.restart(FREE_KICK, 1 - owner.team, tuple(self.players[striker].pos))
            return True
        self.shown.add('foul')
        if kind != 'foul':
            self.shown.add('yellow')
        if kind == 'red':
            self.shown.add('red')
        self.trip(
            self.owner,
            owner.pos[0],
            owner.pos[1],
            {'foul': FOUL, 'yellow': YELLOW_CARD, 'red': RED_CARD}[kind],
        )
        return True

    def step(self):
        """Advances the simulation by one step and returns the game mode of the step."""
        before = list(self.ball)
        for p in self.players:
            p.touch = NO_TOUCH
            if p.action_steps > 0:
                p.action_steps -= 1
                if p.action_steps == 0:
                    p.action = MOVEMENT
        event_mode = self.mode
        if self.mode != NORMAL:
            self.mode_timer -= 1
            if self.mode_timer <= 0:
                self.mode = NORMAL
                self.foul = (-1, -1, NO_FOUL)
                self.offside = -1
                self.pass_ball()
        elif self.scheduled_call():
            pass
        elif self.owner >= 0:
            owner = self.players[self.owner]
            attack = 1.0 if owner.team == 0 else -1.0
            self.ball = [owner.pos[0] + attack * 0.5, owner.pos[1], BALL_RADIUS]
            self.hold += 1
            if self.hold % 3 == 0:
                owner.act(BALL_CONTROL, NON_KICKED)
            if self.hold > self.rng.randint(4, 14):
                self.pass_ball()
        else:
            self.move_ball()
        self.move_players()
        self.roll_ball(before)
        self.step_index += 1
        if not self.half_done and self.step_index >= self.steps // 2:
            self.half_done = True
            self.reset_positions(1)
        return event_mode

    def roll_ball(self, before):
        """Turns the ball by its motion over the step, rolling on the grass and spinning in flight.

        Args:
            before: The ball's position at the start of the step.
        """
        dx, dy = self.ball[0] - before[0], self.ball[1] - before[1]
        distance = math.hypot(dx, dy)
        if distance > MAX_BALL_TRAVEL:
            # A restart puts the ball down at rest.
            self.ball_spin = (0.0, 0.0, 0.0, 1.0)
        elif self.ball[2] <= BALL_RADIUS + 0.01:
            # Rolling without slipping turns the ball about the horizontal axis across its path.
            self.ball_spin = quat_axis_angle((-dy, dx, 0.0), distance / BALL_RADIUS)
        q = quat_mul(self.ball_spin, self.ball_rotation)
        n = math.sqrt(sum(v * v for v in q))
        self.ball_rotation = tuple(v / n for v in q)

    def move_ball(self):
        bx, by, bz = self.ball
        vx, vy, vz = self.ball_vel
        vz -= 9.81 * DT
        bx += vx * DT
        by += vy * DT
        bz += vz * DT
        if bz < BALL_RADIUS:
            bz = BALL_RADIUS
            vz = -vz * 0.5 if abs(vz) > 1.0 else 0.0
        drag = 0.96 if bz <= BALL_RADIUS + 0.01 else 0.99
        self.ball = [bx, by, bz]
        self.ball_vel = [vx * drag, vy * drag, vz]
        if self.keeper_save((bx - vx * DT, by - vy * DT), (bx, by), bz):
            return
        if abs(bx) > PITCH_HALF_W:
            side = 0 if bx < 0 else 1
            if abs(by) < GOAL_HALF_WIDTH and bz < 2.5:
                scorer = 1 - side
                self.score[scorer] += 1
                self.reset_positions(side)
            else:
                if self.kick_team == side:
                    corner_y = math.copysign(PITCH_HALF_H - 0.5, by)
                    corner_x = math.copysign(PITCH_HALF_W - 0.5, bx)
                    self.restart(CORNER, 1 - side, (corner_x, corner_y))
                else:
                    self.restart(GOAL_KICK, side, (math.copysign(PITCH_HALF_W - 6.0, bx), 0.0))
            return
        if abs(by) > PITCH_HALF_H:
            self.restart(THROW_IN, 1 - self.kick_team, (bx, math.copysign(PITCH_HALF_H, by)))
            return
        if math.hypot(vx, vy) < 3.0 and bz < 0.5:
            receiver = min(
                range(len(self.players)),
                key=lambda i: math.hypot(self.players[i].pos[0] - bx, self.players[i].pos[1] - by),
            )
            p = self.players[receiver]
            if math.hypot(p.pos[0] - bx, p.pos[1] - by) < 1.5:
                if receiver != self.offside:
                    self.offside = -1
                if self.rng.random() < 0.04:
                    self.trip(receiver, bx, by)
                    return
                p.act(TRAP if self.rng.random() < 0.3 else BALL_CONTROL, NON_KICKED)
                self.owner = receiver
                self.kick_team = p.team
        for i, p in enumerate(self.players):
            if (
                math.hypot(p.pos[0] - bx, p.pos[1] - by) < 0.8
                and bz < 1.0
                and math.hypot(vx, vy) < 14.0
                and self.rng.random() < 0.3
            ):
                if p.team != self.kick_team:
                    p.act(self.rng.choice((SLIDING, INTERFERE)), NON_KICKED)
                else:
                    p.act(BALL_CONTROL, NON_KICKED)
                self.owner = i
                self.kick_team = p.team
                break

    def trip(self, victim, bx, by, forced=None):
        """An opponent near the receiver trips him, and the referee gives a free kick."""
        p = self.players[victim]
        rivals = [i for i, q in enumerate(self.players) if q.team != p.team and not q.goalkeeper]
        offender = min(
            rivals,
            key=lambda i: math.hypot(self.players[i].pos[0] - bx, self.players[i].pos[1] - by),
        )
        o = self.players[offender]
        # The offender closes in on the victim, so the challenge reads in a replay.
        o.pos = [bx - math.cos(o.heading) * 0.7, by - math.sin(o.heading) * 0.7]
        o.act(SLIDING, ACCIDENTAL)
        p.act(TRIP, NO_TOUCH)
        # The first trip is a plain foul, the next ones show a yellow and a red card first.
        kind = FOUL
        if forced is not None:
            kind = forced
        elif 'foul' not in self.shown:
            self.shown.add('foul')
        elif self.call('yellow', 0.3):
            kind = YELLOW_CARD
        elif self.call('red', 0.04):
            kind = RED_CARD
        if kind == YELLOW_CARD and self.cards[offender] & 2:
            kind = RED_CARD
        if kind == YELLOW_CARD:
            self.cards[offender] |= 2
        elif kind == RED_CARD:
            self.cards[offender] = 2 | 4
        self.foul = (offender, victim, kind)
        self.offside = -1
        self.restart(FREE_KICK, p.team, (bx, by))

    def keeper_save(self, start, end, bz):
        """The keeper of the goal the ball flies at catches or parries it near his line.

        Args:
            start: The ball's position on the pitch before this step, x and y.
            end: The ball's position on the pitch after this step, x and y.
            bz: The ball's height after this step.

        Returns:
            Whether the keeper touched the ball.
        """
        dx, dy = end[0] - start[0], end[1] - start[1]
        if abs(dx) < 0.5 or bz > 2.6:
            return False
        defending = 0 if dx < 0 else 1
        if self.kick_team == defending:
            return False
        for i, p in enumerate(self.players):
            if not p.goalkeeper or p.team != defending:
                continue
            length = dx * dx + dy * dy
            t = ((p.pos[0] - start[0]) * dx + (p.pos[1] - start[1]) * dy) / length
            t = max(0.0, min(1.0, t))
            near = math.hypot(start[0] + t * dx - p.pos[0], start[1] + t * dy - p.pos[1])
            if near > 1.5 or self.rng.random() > 0.45:
                return False
            if self.rng.random() < 0.6:
                p.act(CATCH, NON_KICKED)
                attack = 1.0 if p.team == 0 else -1.0
                self.ball = [p.pos[0] + attack * 0.4, p.pos[1], 1.0]
                self.ball_vel = [0.0, 0.0, 0.0]
                self.owner = i
                self.hold = -8
            else:
                p.act(DEFLECT, KICKED)
                self.ball = [p.pos[0], p.pos[1], max(bz, BALL_RADIUS)]
                self.ball_vel = [-self.ball_vel[0] * 0.35, self.rng.uniform(-9.0, 9.0), 3.0]
            self.kick_team = p.team
            return True
        return False

    def move_players(self):
        bx, by, _ = self.ball
        chasers = set()
        if self.mode == NORMAL and self.owner < 0:
            chasers = {self.nearest(0, self.ball), self.nearest(1, self.ball)}
        elif self.owner >= 0:
            chasers = {self.nearest(1 - self.players[self.owner].team, self.ball)}
        for i, p in enumerate(self.players):
            attack = 1.0 if p.team == 0 else -1.0
            if i == self.owner:
                tx, ty = p.pos[0] + attack * 3.0, p.pos[1] + 0.3 * math.sin(self.step_index / 9)
                top = 6.5
            elif i in chasers:
                tx, ty = bx, by
                top = 7.5
            elif p.goalkeeper:
                gx = -attack * (PITCH_HALF_W - 2.0)
                tx, ty = gx, max(-3.0, min(3.0, by * 0.25))
                top = 4.0
            else:
                shift = bx * 0.55
                tx = p.spot[0] * 0.8 + shift
                ty = p.spot[1] * 0.85 + by * 0.25
                top = 4.5
                owning = self.owner >= 0 and self.players[self.owner].team == p.team
                running = 0 < self.offside_at - self.step_index < 60
                if (owning or running) and p.index == self.per_team - 1:
                    # The striker runs in behind the defence when his team has the ball, and
                    # is caught offside now and then.
                    depths = sorted(
                        (attack * q.pos[0] for q in self.players if q.team != p.team),
                        reverse=True,
                    )
                    tx = attack * min(PITCH_HALF_W - 8.0, depths[1] + 3.0)
                    top = 8.0
            if self.mode == KICK_OFF:
                tx = min(tx, -1.0) if p.team == 0 else max(tx, 1.0)
            dx, dy = tx - p.pos[0], ty - p.pos[1]
            d = math.hypot(dx, dy)
            want = min(top, d * 1.2)
            vx = dx / d * want if d > 1e-6 else 0.0
            vy = dy / d * want if d > 1e-6 else 0.0
            p.vel[0] += (vx - p.vel[0]) * 0.35
            p.vel[1] += (vy - p.vel[1]) * 0.35
            p.pos[0] += p.vel[0] * DT
            p.pos[1] += p.vel[1] * DT
            speed = math.hypot(p.vel[0], p.vel[1])
            if speed > 0.6:
                want_heading = math.atan2(p.vel[1], p.vel[0])
            else:
                want_heading = math.atan2(by - p.pos[1], bx - p.pos[0])
            turn = (want_heading - p.heading + math.pi) % math.tau - math.pi
            p.heading += turn * 0.4
            p.phase += DT * (2.0 + speed * 1.35)
            if p.kick > 0.0:
                p.kick += DT * 3.0
                if p.kick >= 1.0:
                    p.kick = 0.0


def frame_bytes(count):
    """Returns the size in bytes of one frame for count players."""
    return 32 + 28 + 16 + count * 196 + 3 * (((count + 3) // 4) * 4)


def encode_frame(match, mode):
    """Packs the current match state into one frame, in the layout of the rendering guide."""
    count = len(match.players)
    owned_team = match.players[match.owner].team if match.owner >= 0 else -1
    step = match.step_index - 1
    time_ms = int(step * FULL_MATCH_MS / max(1, match.steps - 1))
    flags = 1 if step == match.steps - 1 else 0
    out = bytearray(
        struct.pack(
            '<8i',
            step,
            mode,
            match.score[0],
            match.score[1],
            owned_team,
            match.owner,
            time_ms,
            flags,
        )
    )
    out += struct.pack('<7f', *match.ball, *match.ball_rotation)
    out += struct.pack('<4i', *match.foul, match.offside)
    for p in match.players:
        for node in p.skeleton():
            position = [max(-32767, min(32767, round(value * 500))) for value in node[:3]]
            rotation = [max(-32767, min(32767, round(value * 32767))) for value in node[3:]]
            out += struct.pack('<7h', *position, *rotation)
    padding = bytes(((count + 3) // 4) * 4 - count)
    out += bytes(match.cards) + padding
    out += bytes(p.action for p in match.players) + padding
    out += bytes(p.touch for p in match.players) + padding
    assert len(out) == frame_bytes(count)
    return bytes(out)


def generate_frames(per_team=11, steps=1200, seed=2, names=('Reds', 'Blues'), teams=None):
    """Simulates a synthetic match.

    Args:
        per_team: Players per team, 11 or 5.
        steps: Number of steps, at 10 steps per second.
        seed: Random seed.
        names: Names of the left and right teams.
        teams: Optional header team entries, with name, color, secondary, id and style.

    Returns:
        A pair of the header dict and the list of encoded frames, empty for zero steps.

    Raises:
        ValueError: If steps is negative.
    """
    if steps < 0:
        raise ValueError(f'expected zero or more steps, found {steps}')
    match = Match(per_team, steps, seed, names, teams)
    match.reset_positions(0)
    frames = []
    for _ in range(steps):
        mode = match.step()
        frames.append(encode_frame(match, mode))
    return match.header(), frames


def encode_header(header):
    """Returns the magic, the header length and the header, padded so the frames align to four."""
    blob = json.dumps(header, separators=(',', ':')).encode('utf-8')
    blob += b' ' * (-(len(MAGIC) + 4 + len(blob)) % 4)
    return MAGIC + struct.pack('<I', len(blob)) + blob


def encode_recording(header, frames):
    """Returns the uncompressed recording bytes, magic, header and frames."""
    return encode_header(header) + b''.join(frames)


def build_cjr(per_team=11, steps=1200, seed=2, names=('Reds', 'Blues'), teams=None):
    """Returns a gzip compressed synthetic recording with its header and frames.

    Args:
        per_team: Players per team, 11 or 5.
        steps: Number of steps, at 10 steps per second.
        seed: Random seed.
        names: Names of the left and right teams.
        teams: Optional header team entries, with name, color, secondary, id and style.

    Returns:
        A tuple of the .cjr bytes, the header dict and the list of encoded frames.
    """
    header, frames = generate_frames(per_team, steps, seed, names, teams)
    return gzip.compress(encode_recording(header, frames), 6), header, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('output', help='Path of the .cjr file to write.')
    parser.add_argument('--players', type=int, default=11, choices=(5, 11))
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()
    if args.steps < 0:
        parser.error(f'expected zero or more steps, found {args.steps}')
    data, _, frames = build_cjr(args.players, args.steps, args.seed)
    with open(args.output, 'wb') as f:
        f.write(data)
    score = struct.unpack_from('<2i', frames[-1], 8) if frames else (0, 0)
    print(
        f'wrote {args.output}, {len(frames)} frames, {len(data)} bytes, '
        f'score {score[0]} to {score[1]}'
    )


if __name__ == '__main__':
    main()
