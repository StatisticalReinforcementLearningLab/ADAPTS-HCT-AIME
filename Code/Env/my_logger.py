from . import logargs

# Clear the log file
if logargs.args.logPath is not None:
    with open(logargs.args.logPath, "w") as file:
        pass

class Logger:
    def __init__(self, caller):
        self.log_path = logargs.args.logPath
        self.enabled = logargs.args.logEnabled
        self.caller = caller

    def log(self, message):
        if self.enabled:
            message = f"{self.caller}:\n{message}\n"
            if self.log_path is not None:
                with open(self.log_path, "a") as file:
                    file.write(message + "\n")
            else:
                # print(message)
                pass

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
