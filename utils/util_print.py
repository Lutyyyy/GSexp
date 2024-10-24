from runx.logx import logx

class Bcolors:
    def __init__(self):
        self.HEADER = "\033[95m"
        self.OKBLUE = "\033[94m"
        self.OKGREEN = "\033[92m"
        self.WARNING = "\033[93m"
        self.DARKCYAN = '\033[36m'
        self.DEBUG = "\033[96m"
        self.FAIL = "\033[91m"
        self.ENDC = "\033[0m"

    def disable(self):
        self.HEADER = ""
        self.OKBLUE = ""
        self.OKGREEN = ""
        self.WARNING = ""
        self.DARKCYAN = ""
        self.FAIL = ""
        self.ENDC = ""


def print_grad_stats(grad):
    grad_ = grad.detach()
    print(
        "\nmin, max, mean, std: %e, %e, %e, %e"
        % (
            grad_.min().item(),
            grad_.max().item(),
            grad_.mean().item(),
            grad_.std().item(),
        )
    )


bcolors = Bcolors()
# bcolors.disable()
STR_STAGE = bcolors.OKBLUE + "==>" + bcolors.ENDC
STR_VERBOSE = bcolors.OKGREEN + "[Verbose]" + bcolors.ENDC
STR_INFO = bcolors.DARKCYAN + "INFO: " + bcolors.ENDC
STR_WARNING = bcolors.WARNING + "[Warning]" + bcolors.ENDC
STR_ERROR = bcolors.FAIL + "[Error]" + bcolors.ENDC
STR_DEBUG = bcolors.DEBUG + "[Debug...]" + bcolors.ENDC
logx.initialize(logdir='./trash/test_runx', coolname=False, tensorboard=True)