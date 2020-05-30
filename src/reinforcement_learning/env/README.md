## Important installations:

### For core gym environments:
```sh
~/anaconda3/envs/gym_env/bin/pip install box2d-py
~/anaconda3/envs/gym_env/bin/pip install pyglet==v1.3.2 (pyglet-1.5.5 for simple simulation)
```

### For custom gym environments (added via submodules to the project):
```sh
~/anaconda3/envs/gym_env/bin/pip install . (in submodule directories for installation of the third party environments)
```

Also probably you will need to mark gym-duckietown as a source root

### Some words about DonkeyCar simulation:
In order to have full functionality and ability to change parameters of the virtual camera or 
change scenes and models you need to have [Unity](https://unity.com/) and [sdsandbox](https://github.com/tawnkramer/sdsandbox) installed. 

But if you only want to receive the observations from the environment and apply some actions you need just a binary file 
(compiled Unity project). No need to have Unity installed in this case.