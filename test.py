import local_dmc2gym as dmc2gym
import cv2
from video import VideoRecorder

domain_name = 'cheetah'
task_name = 'run'
distract_type = 'dots'
difficulty = 'hard'
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

save_dir = './test/'
video_recorder = VideoRecorder(save_dir)
video_recorder.init()


def main():
	cv2.namedWindow("env", 0)
	cv2.resizeWindow("env", 1000, 1000)
	i = 0
	while True:
		env.reset()
		done = False
		while not done:
			img = env.render(mode='rgb_array')
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			action = env.action_space.sample()
			obs, reward, done, info = env.step(action)
			video_recorder.record(env)
			cv2.imshow('env', img)
			if cv2.waitKey(100) & 0xFF == ord('q'):
				return
			i += 1
			# if i == 200:
			# 	video_recorder.save('RandomDots_for.mp4')
			# 	return


if __name__ == '__main__':
	main()
	print('done')

	