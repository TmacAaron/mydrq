import local_dmc2gym as dmc2gym
import cv2

domain_name = 'cheetah'
task_name = 'run'
distract_type = 'dots'
difficulty = 'easy'
ground = 'forground'
background_dataset_path = None
seed = 1
image_size = 256
action_repeat = 1

env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       distract_type=distract_type,
                       ground=ground,
                       difficulty=difficulty,
                       background_dataset_path=background_dataset_path,
                       seed=seed,
                       visualize_reward=False,
                       from_pixels=True,
                       height=image_size,
                       width=image_size,
                       frame_skip=action_repeat,
                       )

def main():
	cv2.namedWindow("env", 0)
	cv2.resizeWindow("env", 1000, 1000)
	while True:
		env.reset()
		done = False
		i = 0
		while not done:
			img = env.render(mode='rgb_array', height=256, width=256)
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			action = env.action_space.sample()
			obs, reward, done, info = env.step(action)

			cv2.imshow('env', img)
			i += 1
			if cv2.waitKey(20) & 0xFF == ord('q'):
				return
		print(i)

if __name__ == '__main__':
	main()

	