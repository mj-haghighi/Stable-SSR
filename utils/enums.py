class RelabelingStrategy:
    BASE_LINE = "base_line"
    STABLE = "stable"

class SamplingStrategy:
    BASE_LINE = "base_line"
    STABLE = "stable"
    STABLE_EXTENDED_EXCLUDED = "stable_extended_excluded"

class DATASETS:
    CIFAR10     = "cifar10"
    CIFAR100    = "cifar100"
    ANIMAL10N   = "animal10n"
    CIFAR10NAG  = "cifar10nag"
    CIFAR10NWS  = "cifar10nws"

class NOISE_MODE:
    AGGRE = "aggre"
    WORSE = "worse"
    SYM = "sym"
    ASYM = "asym"