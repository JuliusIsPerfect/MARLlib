from marllib import marl

'''
env = marl.make_env(environment_name="mate", map_name="MATE-4v2-9-v0", coop_team="target")
# env = marl.make_env(environment_name="mate", map_name="MATE-4v2-9-v0", coop_team="camera")
# initialize algorithm with appointed hyperparameters
mappo = marl.algos.mappo(hyperparam_source="common")
# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "8-8"})
# start training
mappo.fit(env, model, stop={"timesteps_total": 1000000}, checkpoint_freq=10, local_mode=True,
          num_gpus=0, num_workers=2, share_policy="group")

'''
env = marl.make_env(environment_name="mate", map_name="MATE-4v2-9-v0")
algo = marl.algos.mappo(hyperparam_source="test")
model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
         num_workers=2, share_policy="group", checkpoint_end=False)
