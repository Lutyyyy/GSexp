class Bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    DARKCYAN = "\033[36m"
    DEBUG = "\033[96m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"

    @classmethod
    def disable(cls):
        for attr in vars(cls):
            if not attr.startswith("__"):
                setattr(cls, attr, "")


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


# Bcolors.disable()
STR_STAGE = f"{Bcolors.OKBLUE}==>{Bcolors.ENDC}"
STR_VERBOSE = f"{Bcolors.OKGREEN}[Verbose]{Bcolors.ENDC}"
STR_INFO = f"{Bcolors.DARKCYAN}INFO: {Bcolors.ENDC}"
STR_WARNING = f"{Bcolors.WARNING}[Warning]{Bcolors.ENDC}"
STR_ERROR = f"{Bcolors.FAIL}[Error]{Bcolors.ENDC}"
STR_DEBUG = f"{Bcolors.DEBUG}[Debug...]{Bcolors.ENDC}"
