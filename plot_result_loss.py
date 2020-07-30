import matplotlib.pyplot as plt
from collections import defaultdict
import pathlib
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", required=True, type=str, help="path of the result folder.")
    parser.add_argument("--nrows", type=int, default = 2, help="number of rows for the plot.")
    parser.add_argument("--log_scale", action='store_true', default = False, help="Draw y axis in log scale.")

    return parser.parse_args()

class LossParser:

    data_to_ignore = ('timestamp', 'tag')

    def __init__(self, args):
        self.result_dir = args.result_path
        self.loss_path = pathlib.Path(self.result_dir, 'losses.txt')
        assert self.loss_path.is_file(), "'losses.txt' does not exist in {}".format(self.result_dir)

    def parse(self):
        with open(str(self.loss_path), 'r') as f:
            lines = f.readlines()

        parsed_data = defaultdict(list)

        for line in lines:
            line_data = self._parse_line(line)
            for k, v in line_data.items():
                parsed_data[k].append(v)

        return parsed_data

    def _parse_line(self, line):
        line = line.strip()
        data_str = line.split(',')
        data_str = self._remove_data_to_ignore(data_str)
        data = {}
        for d in data_str:
            k, v = d.split(':')
            data[k] = float(v)

        return data

    def _remove_data_to_ignore(self, data_str):
        filtered = []
        for d in data_str:
            if not d.startswith(self.data_to_ignore):
                filtered.append(d)
        return filtered

class Plotter:
    data_type_to_not_plot = ('epoch', 'iter')
    def __init__(self, data, args):
        self.n_row = args.nrows
        self.log_scale = args.log_scale
        self.data = data

    def plot(self):
        self.data = self._change_iter_to_epoch(self.data)
        self._plot_in_subplots(self.data)

        plt.show()

    def _change_iter_to_epoch(self, data):
        iter_per_epoch = self._count_iter_per_epoch(data)
        new_iter = []
        for i, iter in enumerate(data['iter']):
            new_iter.append((i+1) * iter_per_epoch)

        data['iter'] = new_iter
        return data

    def _count_iter_per_epoch(self, data):
        epoch_set = set()
        for epoch in data['epoch']:
            epoch_set.add(epoch)

        num_epoch = len(epoch_set)
        num_iter = len(data['iter'])

        return num_epoch / num_iter

    def _plot_in_subplots(self, data):
        data_type_to_plot = self._get_data_type_to_plot(data)
        fig, axises = self._create_subplots(data_type_to_plot)

        x = data['iter']
        for i, data_type in enumerate(data_type_to_plot):
            y = data[data_type]
            self._plot_data_in_ax(axises[i], x, y, data_type)

    def _create_subplots(self, data_types):
        n_col = self._get_n_col(data_types)
        fig, ax = plt.subplots(nrows = self.n_row, ncols = n_col)

        flatten_ax = []
        for row in ax:
            for col in row:
                flatten_ax.append(col)

        return fig, flatten_ax

    def _get_data_type_to_plot(self, data):
        filtered = []
        for key in data.keys():
            if not key in self.data_type_to_not_plot:
                filtered.append(key)

        return filtered

    def _get_n_col(self, data_types):
        n_cols = int(len(data_types) / self.n_row)
        if len(data_types) % self.n_row > 0:
            n_cols += 1

        return n_cols

    def _plot_data_in_ax(self, ax, x, y, title):
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        if self.log_scale:
            ax.set_yscale('log')
        ax.plot(x,y)

def main(args):

    loss_parser =  LossParser(args)
    parsed_data = loss_parser.parse()

    plotter = Plotter(parsed_data, args)
    plotter.plot()

if __name__ == "__main__":
    args = parse_args()
    main(args)
