from abc import ABC, abstractmethod

class BaseOption(ABC):
    """ This is basic abstract class for options. """
    def __init__(self, name, prefix):
        self.__name__ = name
        self.__prefix__ = prefix
        self.arguments = []
        self.set_arguments()
        self.initialized = False

    @abstractmethod
    def set_arguments(self):
        """ Set (custom) arguments that this option needs. """
        pass

    def add_to_parser(self, parser):
        for argument in self.arguments:
            argument.add_to_parser(parser, self.__prefix__)

    def initialize(self, args):
        if self.initialized:
            return

        for argument in self.arguments:
            arg_name = argument.update_arg_name(self.__prefix__)
            v = getattr(args, arg_name)
            setattr(self, argument.name, v)

        del self.arguments
        self.initialized = True

    def initialize_with_defaults(self):
        if self.initialized:
            return

        for argument in self.arguments:
            setattr(self, argument.name, argument.default)

        del self.arguments
        self.initialized = True

    def __str__(self):
        message = ""
        for k, v in sorted(vars(self).items()):
            message += "{}: {}\n".format(str(k), str(v))

        return message

    def convert_mode_str_to_mode(self):
        if hasattr(self, 'mode'):
            self.mode = Mode(self.mode)


class Argument:
    """ This is a wrapper class for python argument. """
    def __init__(self, name, type=None, default=None, help=None, action=None):
        self.name = name
        self.type = type
        self.default = default
        self.help = help
        self.action = action

    def add_to_parser(self, parser, prefix):
        parser_tag = self.get_parser_tag(prefix)
        helper_msg = self.update_helper_msg(prefix)
        if not self.action is None:
            self._add_action_to_parser(parser, parser_tag, prefix)
            return

        parser.add_argument(parser_tag, \
                            type = self.type, \
                            default = self.default, \
                            help = helper_msg, \
                            action = self.action \
                            )

    def _add_action_to_parser(self, parser, parser_tag, prefix):
        parser.add_argument(parser_tag, \
                            default = self.default, \
                            help = self.help, \
                            action = self.action \
                            )

    def get_parser_tag(self, prefix):
        return "--" + self.update_arg_name(prefix)

    def update_arg_name(self, prefix):
        if prefix:
            return '{}_{}'.format(prefix, self.name)
        else:
            return '{}'.format(self.name)

    def update_helper_msg(self, prefix):
        if prefix:
            return '{}: {}'.format(prefix, self.help)
        else:
            return '{}'.format(self.help)

class Mode:
    """ Mode class defines the mode of models and networks.
        A model or a network might have to work differently for different modes.
        For example, a network might have to calculate intermediate losses for 'train' and 'test', but not for 'eval'.
        At the sametime, the network should be swtiched to eval mode for 'test' and 'eval', but not for 'train'.

        There are three modes.
        - train
        - test
        - eval (evaluate)
    """
    modes = ['train', 'test', 'eval']
    def __init__(self, mode_str):
        self.mode_str = self.clean_up(mode_str)
        assert self.mode_str in self.modes, "Mode: mode should be one of {}".format(','.join(modes))

    def clean_up(self, mode_str):
        mode_str = mode_str.strip()
        mode_str = mode_str.lower()

        return mode_str

    def is_train(self):
        return self.mode_str == 'train'

    def is_test(self):
        return self.mode_str == 'test'

    def is_eval(self):
        return self.mode_str == 'eval'

    def __str__(self):
        return "{} (mode class)".format(self.mode_str)

    def to_train(self):
        self.mode_str = 'train'

    def to_test(self):
        self.mode_str = 'test'

    def to_eval(self):
        self.mode_str = 'eval'
