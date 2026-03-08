from setuptools import setup
from setuptools.command.editable_wheel import editable_wheel


class CompatEditableWheel(editable_wheel):
    def run(self):
        self.mode = "compat"
        super().run()


setup(cmdclass={"editable_wheel": CompatEditableWheel})
