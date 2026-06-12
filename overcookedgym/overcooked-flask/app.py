import os
import json
import copy
import argparse

from flask import Flask, jsonify, redirect, request, send_file

app = Flask(__name__)


def get_prediction(s, policy, layout_name, algo):
    s = torch.tensor(s).unsqueeze(0).float()
    # print(s.size())
    actions, states = policy.predict(observation=s)
    # print(actions)
    return int(actions[0])


def process_state(state_dict, layout_name):
    def object_from_dict(object_dict):
        return ObjectState(**object_dict)

    def player_from_dict(player_dict):
        held_obj = player_dict.get("held_object")
        if held_obj is not None:
            player_dict["held_object"] = object_from_dict(held_obj)
        return PlayerState(**player_dict)

    def state_from_dict(state_dict):
        state_dict["players"] = [player_from_dict(
            p) for p in state_dict["players"]]
        object_list = [object_from_dict(o)
                       for _, o in state_dict["objects"].items()]
        state_dict["objects"] = {ob.position: ob for ob in object_list}
        return OvercookedState(**state_dict)

    state = state_from_dict(copy.deepcopy(state_dict))
    print(state.to_dict())

    return MDP.featurize_state(state, MLP)


def convert_traj_to_simultaneous_transitions(traj_dict, layout_name):

    ego_obs = []
    alt_obs = []
    ego_act = []
    alt_act = []
    flags = []

    for state_list in traj_dict['ep_states']:  # loop over episodes
        ego_obs.append([process_state(state, layout_name)[0]
                        for state in state_list])
        alt_obs.append([process_state(state, layout_name)[1]
                        for state in state_list])

        # check pantheonrl/common/wrappers.py for flag values
        flag = [0 for state in state_list]
        flag[-1] = 1
        flags.append(flag)

    for action_list in traj_dict['ep_actions']:  # loop over episodes
        ego_act.append([joint_action[0] for joint_action in action_list])
        alt_act.append([joint_action[1] for joint_action in action_list])

    ego_obs = np.concatenate(ego_obs, axis=-1)
    alt_obs = np.concatenate(alt_obs, axis=-1)
    ego_act = np.concatenate(ego_act, axis=-1)
    alt_act = np.concatenate(alt_act, axis=-1)
    flags = np.concatenate(flags, axis=-1)

    return SimultaneousTransitions(
            ego_obs,
            ego_act,
            alt_obs,
            alt_act,
            flags,
        )


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data_json = json.loads(request.data)
        state_dict, player_id_dict, server_layout_name, algo, timestep = data_json["state"], data_json[
            "npc_index"], data_json["layout_name"], data_json["algo"], data_json["timestep"]
        player_id = int(player_id_dict)
        layout_name = NAME_TRANSLATION[server_layout_name]
        s0, s1 = process_state(state_dict, layout_name)

        # print(s0.to_dict())
        # print(s1.to_dict())
        print("---\n")

        if ARGS.replay_traj:
            if player_id == 0:
                a = int(EGO_TRANSITIONS.acts[timestep][0]) if timestep < len(
                    EGO_TRANSITIONS.acts) else 4
            elif player_id == 1:
                a = int(ALT_TRANSITIONS.acts[timestep][0]) if timestep < len(
                    EGO_TRANSITIONS.acts) else 4
            else:
                assert(False)
        else:
            if player_id == 0:
                s, policy = s0, POLICY_P0
            elif player_id == 1:
                s, policy = s1, POLICY_P1
            else:
                assert(False)
            a = get_prediction(s, policy, layout_name, algo)

        print(a)
        # print(algo)
        print("sending action ", a)
        return jsonify({'action': a})


@app.route('/updatemodel', methods=['POST'])
def updatemodel():
    if request.method == 'POST':
        data_json = json.loads(request.data)
        traj_dict, traj_id, server_layout_name, algo = data_json["traj"], data_json[
            "traj_id"], data_json["layout_name"], data_json["algo"]
        layout_name = NAME_TRANSLATION[server_layout_name]
        print(traj_id)

        if ARGS.trajs_savepath:
            # Save trajectory (save this to keep reward information)
            filename = "%s.json" % (ARGS.trajs_savepath)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(traj_dict, f)

            # Save transitions minimal (only state/action/done, no reward)
            simultaneous_transitions = convert_traj_to_simultaneous_transitions(
                traj_dict, layout_name)
            simultaneous_transitions.write_transition(ARGS.trajs_savepath)

        # Finetune model: todo

        REPLAY_TRAJ_IDX = 0
        done = True
        return jsonify({'status': done})


@app.route('/')
def root():
    if ARGS.replay_json:
        return redirect(
            '/replay?trajectory=/trajectory&autoplay=1&step_ms=150'
            '&server_file=1'
        )
    return app.send_static_file('index.html')


@app.route('/replay')
def replay():
    return app.send_static_file('replay.html')


@app.route('/trajectory')
def trajectory():
    if not ARGS.replay_json:
        return jsonify({'error': 'No --replay_json file was configured'}), 404
    return send_file(ARGS.replay_json, mimetype='application/json')


@app.route('/trajectory-info')
def trajectory_info():
    if not ARGS.replay_json:
        return jsonify({'error': 'No --replay_json file was configured'}), 404
    return jsonify({'path': ARGS.replay_json})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--modelpath_p0', type=str,
                        help="path to load model for player 0")
    parser.add_argument('--modelpath_p1', type=str,
                        help="path to load model for player 1")
    parser.add_argument('--replay_traj', type=str,
                        help="replay traj, don't run policies")
    parser.add_argument('--replay_json', type=str,
                        help="browser replay JSON exported from HARL")
    parser.add_argument('--layout_name', type=str,
                        help="layout name")
    parser.add_argument('--trajs_savepath', type=str,
                        help="path to save trajectories")
    parser.add_argument('--host', default='0.0.0.0',
                        help="host interface for the Flask server")
    parser.add_argument('--port', type=int, default=5000,
                        help="port for the Flask server")
    parser.add_argument('--debug', action='store_true',
                        help="enable Flask debug mode and auto-reloading")
    ARGS = parser.parse_args()

    if ARGS.replay_json:
        if ARGS.modelpath_p0 or ARGS.modelpath_p1 or ARGS.replay_traj:
            parser.error(
                '--replay_json cannot be combined with models or --replay_traj')
        ARGS.replay_json = os.path.abspath(ARGS.replay_json)
        if not os.path.isfile(ARGS.replay_json):
            parser.error('replay JSON does not exist: %s' % ARGS.replay_json)
    elif not ARGS.layout_name:
        parser.error('--layout_name is required unless --replay_json is used')
    else:
        import gym
        import numpy as np
        import torch
        from overcooked_ai_py.mdp.overcooked_mdp import (
            ObjectState,
            OvercookedGridworld,
            OvercookedState,
            PlayerState,
        )
        from overcooked_ai_py.planning.planners import (
            MediumLevelPlanner,
            NO_COUNTERS_PARAMS,
        )
        from stable_baselines3 import PPO

        from overcookedgym.overcooked_utils import NAME_TRANSLATION
        from pantheonrl.common.trajsaver import SimultaneousTransitions

        if ARGS.replay_traj:
            assert(not ARGS.modelpath_p0 and not ARGS.modelpath_p1)
            env = gym.make('OvercookedMultiEnv-v0', layout_name=ARGS.layout_name)
            simultaneous_transitions = SimultaneousTransitions.read_transition(
                "%s.npy" % ARGS.replay_traj,
                env.observation_space,
                env.action_space,
            )
            EGO_TRANSITIONS = simultaneous_transitions.get_ego_transitions()
            ALT_TRANSITIONS = simultaneous_transitions.get_alt_transitions()
        else:
            # at least one policy should be specified, the other can be human
            assert(ARGS.modelpath_p0 or ARGS.modelpath_p1)
            if ARGS.modelpath_p0:
                POLICY_P0 = PPO.load(ARGS.modelpath_p0)
            if ARGS.modelpath_p1:
                POLICY_P1 = PPO.load(ARGS.modelpath_p1)

        # TODO: client should pick layout name, instead of server?
        # currently both client/server pick layout name, and they must match
        MDP = OvercookedGridworld.from_layout_name(layout_name=ARGS.layout_name)
        MLP = MediumLevelPlanner.from_pickle_or_compute(
            MDP, NO_COUNTERS_PARAMS, force_compute=False)

    print(
        "Starting Overcooked replay viewer on "
        "http://{0}:{1}/".format(ARGS.host, ARGS.port),
        flush=True,
    )
    app.run(
        debug=ARGS.debug,
        host=ARGS.host,
        port=ARGS.port,
        use_reloader=ARGS.debug,
    )
