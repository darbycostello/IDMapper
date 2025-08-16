class ProgressCM:

	def __init__(self, wm=None, steps=100):
		self.running = False
		self.wm = wm
		self.steps = steps
		self.current_step = 0

	def __enter__(self):
		if self.wm:
			self.wm.progress_begin(0, self.steps)
			self.step()
			self.running = True
		return self

	def __exit__(self, *args):
		self.running = False
		if self.wm:
			self.wm.progress_end()
			self.wm = None
		else:
			print("Done.\n")

	def step(self, amount=1):
		self.current_step += amount
		self.current_step = min(self.current_step, self.steps)
		if self.wm:
			self.wm.progress_update(self.current_step)
		else:
			print("Step "
					+ str(self.current_step)
					+ "/" + str(self.steps))
