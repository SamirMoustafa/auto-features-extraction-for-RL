import pygame
import threading
import numpy as np


class Teleoperator:
    def __init__(self, env, shared_dict, action_queue):
        self.env_ = env
        self.action_space_ = self.env_.action_space
        self.shared_dict_ = shared_dict
        self.action_queue_ = action_queue
        self.process_ = None
        self.clock = pygame.time.Clock()

        self.x_ = 0.0
        self.theta_ = 0.0
        self.control_throttle_ = 0.0
        self.control_steering_ = 0.0

    def main_loop(self):
        pygame.init()
        pygame.display.set_mode((200, 200))

        while True:
            end = False
            self.theta_, self.x_ = 0, 0
            while not end:
                pygame_key = pygame.key.get_pressed()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        end = True
                        pygame.quit()
                    if event.type == pygame.KEYDOWN:
                        end = True
                        if event.key == pygame.K_m:
                            self.shared_dict_["manual_mode"] = not self.shared_dict_["manual_mode"]
                            print("Manual mode:" + str(self.shared_dict_["manual_mode"]))
                        elif event.key == pygame.K_e:
                            self.shared_dict_["exploration_mode"] = False

                self.process_key(pygame_key)

                #pygame.time.Clock().tick(60.0)

    def process_key(self, pygame_key):

        if self.shared_dict_["manual_mode"] is False:
            return

        control_key = False
        if pygame_key[pygame.K_UP]:
            self.x_ += 0.02
            control_key = True
        elif pygame_key[pygame.K_DOWN]:
            self.x_ -= 0.02
            control_key = True
        else:
            self.x_ = 0
            control_key = True
        if pygame_key[pygame.K_LEFT]:
            self.theta_ -= 0.02
            control_key = True
        elif pygame_key[pygame.K_RIGHT]:
            self.theta_ += 0.02
            control_key = True
        else:
            self.theta_ = 0
            control_key = True
        if control_key:
            # self.control_throttle_, self.control_steering_ = self.control(self.x_, self.theta_,
            #                                                                self.control_throttle_,
            #                                                                self.control_steering_)
            # self.action_queue_.put([self.control_steering_, self.control_throttle_])
            # self.shared_dict_["action"] = [self.control_steering_, self.control_throttle_]
            action = np.clip([self.theta_, self.x_], self.action_space_.low, self.action_space_.high)
            self.shared_dict_["action"] = action

    def control(self, x, theta, control_throttle, control_steering):
        """
        Smooth control.
        :param x: (float)
        :param theta: (float)
        :param control_throttle: (float)
        :param control_steering: (float)
        :return: (float, float)
        """

        MAX_STEERING = 1
        MIN_STEERING = - MAX_STEERING

        # Simulation config
        MIN_THROTTLE = 0.4
        # max_throttle: 0.6 for level 0 and 0.5 for level 1
        MAX_THROTTLE = 0.6

        MAX_TURN = 1
        # Smoothing constants
        STEP_THROTTLE = 0.3
        STEP_TURN = 0.4

        target_throttle = x
        target_steering = MAX_TURN * theta
        if target_throttle > control_throttle:
            control_throttle = min(target_throttle, control_throttle + STEP_THROTTLE)
        elif target_throttle < control_throttle:
            control_throttle = max(target_throttle, control_throttle - STEP_THROTTLE)
        else:
            control_throttle = target_throttle

        if target_steering > control_steering:
            control_steering = min(target_steering, control_steering + STEP_TURN)
        elif target_steering < control_steering:
            control_steering = max(target_steering, control_steering - STEP_TURN)
        else:
            control_steering = target_steering
        return control_throttle, control_steering

    def start(self):
        self.process_ = threading.Thread(target=self.main_loop)
        # Make it a deamon, so it will be deleted at the same time
        # of the main process
        self.process_.daemon = True
        self.process_.start()
