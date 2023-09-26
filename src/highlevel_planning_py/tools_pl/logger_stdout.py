class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class LoggerStdout:
    def __init__(self, level="info"):
        self.level = level

    @staticmethod
    def warning(input_str):
        print(f"{bcolors.WARNING}{input_str}{bcolors.ENDC}")

    def info(self, input_str):
        if self.level != "info":
            return
        print(input_str)

    @staticmethod
    def error(input_str):
        print(f"{bcolors.FAIL}{input_str}{bcolors.ENDC}")
