import itertools
import pandas as pd
import random

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import Counter, defaultdict


# ----
# Helper functions
# ----

VERBOSE = False  # Set true to get all printouts

def print_update_header(destination, location):
    """Prints destination and location. Helpful to keep track of location for each
    update"""
    if not VERBOSE: return
    print "-" * 40
    print "destination =", destination
    print "location =", location


def print_state_qvals(qmatrix, state):
    """Prints out the values of Q for all actions in the given state"""
    if not VERBOSE: return
    print {k: round(val, 2) for k, val in qmatrix[state].items()}


def print_transition(inputs, deadline, waypoint, action, reward):
    """
    This function provides an enhanced print out for more easily keeping track
    of the simulation from the perspective of our agent.  It prints out an
    ASCII representation of the intersection along with the following values:

    - next_waypoint
    - action
    - reward

    """
    if not VERBOSE: return

    def first_leter(direction):
        return ' ' if inputs[direction] is None else inputs[direction][0]

    intersection = ('   [%s]\n[%s] %s [%s]' %
                    (first_leter('oncoming'), first_leter('left'), first_leter('light'), first_leter('right')))

    print 'next_waypoint =', waypoint
    print 'deadline =', deadline
    print intersection
    print 'action = ', action
    print 'reward = ', reward

    return waypoint, intersection, action, reward


def print_qvalue_update(starting_q_value, updated_q_value):
    """Prints the before and after values of the Q element as given"""
    if not VERBOSE: return

    print "starting Q value = %.2f" % starting_q_value
    print " updated Q value = %.2f" % updated_q_value


def make_q_matrix(actions):
    """
    We represent the Q matrix as a dict of dicts.  The keys of the top level
    dict are states, and the values are dicts of action:Qvalue pairs:

        Q = {state: {action: Qval}}

    Q is initialized with random values in the interval [0, 1] for every
    possible action.

    """
    return defaultdict(lambda: {action: random.random() for action in actions})


# ----
# Agent classes
# ----

class RandomAgent(Agent):
    """An agent that takes random actions in the smartcab world."""

    def __init__(self, env):
        super(RandomAgent, self).__init__(
            env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.transitions = Counter()

    def reset(self, destination=None):
        self.planner.route_to(destination)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Select action randomly:
        action = random.choice([None, 'forward', 'left', 'right'])

        # Execute action and get reward:
        reward = self.env.act(self, action)

        transition = print_transition(inputs, deadline, self.next_waypoint, action, reward)
        self.transitions[transition] += 1

    def set_q_params(self, alpha_0, theta, s):
        pass


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(
            env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        self.possible_actions = [None, 'forward', 'left', 'right']
        self.Q = make_q_matrix(self.possible_actions)

        # Q-learning parameter
        self.discount = 0.1

        # List for collecting the results of each simulation trial
        self.simulation_results = list()

    def set_q_params(self, alpha_0, theta, s):
        self.alpha_0 = alpha_0
        self.theta = theta
        self.s = s

    def learning_rate(self):
        """
        The learning rate must decrease with time for convergence of the Q matrix.

        Here I choose a 1/t dependency, but offset in time such that the learning
        rate does not decay too quickly.

        The starting value of the learning rate is 0.1 and by the 10000th transition
        it will have decreased to 0.05.
        """
        starting_value = self.alpha_0
        offset = self.theta
        return starting_value * offset / (offset + self.num_updates)

    def epsilon(self):
        """
        Epsilon is the exploration parameter, and similar to the learning rate we
        want it to decrease as the agent learns the driving rules. We choose the
        exact same dependency as the learning rate, but scale it down by a factor
        so that there is only a small chance of taking a random action.
        """
        scale_down = self.s
        return self.learning_rate() / scale_down

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.total_reward = 0
        self.num_updates = 0
        self.num_red_lights = 0  # num. of steps with red light
        self.num_can_move = 0  # num. of steps in which smartcab can move (green light, or red light right turn)
        self.rewards_counter = Counter()

        location = self.env.agent_states[self]['location']
        self.destination = destination

        # Determine the distance to the destination, considering that the
        # smartcab can wrap around the grid:
        xdist = min(abs(location[0] - destination[0] + 8 * i) for i in [-1, 0, 1])
        ydist = min(abs(location[1] - destination[1] + 6 * i) for i in [-1, 0, 1])
        self.distance_to_destination = xdist + ydist

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        state = self.get_state_tuple(inputs, self.next_waypoint)

        # TODO: Select action according to your policy
        print_update_header(self.destination, self.env.agent_states[self]['location'])
        print_state_qvals(self.Q, state)
        action = self.argmax_q(state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        print_transition(inputs, deadline, self.next_waypoint, action, reward)

        # TODO: Learn policy based on state, action, reward
        new_state = self.get_state_tuple(self.env.sense(self), self.planner.next_waypoint())
        discounted_reward = reward + self.discount * self.max_q(new_state)

        starting_q_value = self.Q[state][action]
        updated_q_value = ((1 - self.learning_rate()) * starting_q_value
                           + self.learning_rate() * discounted_reward)
        self.Q[state][action] = updated_q_value
        print_qvalue_update(starting_q_value, updated_q_value)

        # Update monitoring variables
        self.total_reward += reward
        self.num_updates += 1
        if inputs['light'] == 'red':
            self.num_red_lights += 1
        if inputs['light'] == 'green' or (inputs['light'] == 'red' and self.next_waypoint == 'right'):
            self.num_can_move += 1
        self.rewards_counter[reward] += 1

        is_run_finished = self.env.done or deadline <= 0
        if is_run_finished:
            num_traffic_violations = self.rewards_counter[-0.5]
            result = {
                'success': self.env.done,
                'total_reward': self.total_reward,
                'num_updates': self.num_updates,
                'num_red_lights': self.num_red_lights,
                'num_can_move': self.num_can_move,
                'distance': self.distance_to_destination,
                'violations': num_traffic_violations,  # a reward of -0.5 is given when there is a traffic violation
                'effective_speed':  float(self.distance_to_destination) / float(self.num_can_move + num_traffic_violations)
            }
            result.update(self.rewards_counter)
            self.simulation_results.append(result)

    @staticmethod
    def get_state_tuple(inputs, waypoint):
        """Returns a tuple that represents the state given the current inputs and waypoint"""
        is_light_green = inputs['light'] == 'green'
        car_oncoming_turning = inputs['oncoming'] == 'left' or inputs['oncoming'] == 'right'
        car_oncoming = inputs['oncoming'] is not None
        car_left = inputs['left'] is not None
        car_right = inputs['right'] is not None
        return is_light_green, waypoint, car_oncoming_turning, car_oncoming, car_left, car_right

    def argmax_q(self, state):
        """Returns the action that maximizes Q for the given state"""
        maxQ = None
        argmax = None
        for ii, (action, Qval) in enumerate(self.Q[state].items()):
            if Qval > maxQ or ii == 0:
                maxQ = Qval
                argmax = action

        # We select a random exploration choice with probability epsilon:
        if random.random() < self.epsilon():
            return random.choice(self.possible_actions)
        else:
            return argmax

    def max_q(self, state):
        """Returns the value of Q maximized over all possible actions for the given state"""
        return max(self.Q[state].values())


# ----
# Run simulations
# ----

def run():
    """Run the agent for a finite number of trials."""

    # Iterate over the agent Q-learning paramters:
    grid_search_results = []
    # for alpha_0, theta, S in itertools.product([0.05, 0.1, 0.5, 1., 2], [1e2, 1e4, 1e6], [1, 10, 100]):
    for alpha_0, theta, S in itertools.product([0.1], [1e4], [10]):
        # Set up environment and agent
        e = Environment(num_dummies=3)  # create environment (also adds some dummy traffic)

        agent_class = LearningAgent  # choose from: (RandomAgent, LearningAgent)

        a = e.create_agent(agent_class)  # create agent

        e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
        # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

        a.set_q_params(alpha_0, theta, S)

        # Now simulate it
        sim = Simulator(e, update_delay=0.0001, display=False)  # create simulator (uses pygame when display=True, if available)
        # NOTE: To speed up simulation, reduce update_delay and/or set display=False

        sim.run(n_trials=100)  # run for a specified number of trials
        # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

        if isinstance(a, RandomAgent):
            # If we are running with the random agent, go ahead and save all
            # of the transitions that were observed. The results can be inspected
            # in an interactive session.
            import pickle
            pickle.dump(a.transitions, open('random_transitions.pck', 'wb'))
            break  # the Q-learning params have no effect on the RandomAgent, so no need to go through with the grid search.

        if isinstance(a, LearningAgent):
            # If we are running with the learning agent, print out some statistics
            # over the total number of simulation trials.
            df = pd.DataFrame(a.simulation_results)
            df = df.fillna(0.)  # Fill values for reward statistics

            total_penalties = float(sum(df[-0.5]) + sum(df[-1.0]))
            total_updates = float(sum(df.num_updates))
            success_rate = float(sum(df.success)) / float(len(df))
            penalty_rate = total_penalties / total_updates

            grid_search_results.append({
                'alpha_0': alpha_0,
                'theta': theta,
                'S': S,
                'success_rate': success_rate,
                'penalty_rate': penalty_rate,
                'objective': success_rate - 2. * penalty_rate
            })

    order = ['alpha_0', 'theta', 'S', 'success_rate', 'penalty_rate', 'objective']
    grid_df = pd.DataFrame(grid_search_results).sort_values(by='objective', ascending=False)[order]

    import StringIO
    output = StringIO.StringIO()

    output.write("Top 5: \\newline\n")
    grid_df.head().to_latex(output, index=False)
    output.write("\\vspace{1em} Worse 5: \\newline\n")
    grid_df.tail().to_latex(output, index=False)
    contents = output.getvalue()
    print contents.replace('toprule', 'hline').replace('midrule', 'hline').replace('bottomrule', 'hline')

if __name__ == '__main__':
    run()
