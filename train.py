from option.options import Options
from preset import modify_options
from dataset import *
from run import find_run_using_name

def iter_to_epoch(cur_iter, batch_size, num_data):
    print(cur_iter, batch_size, num_data)
    return float(cur_iter * batch_size) / num_data

def epoch_to_iter(epoch, total_epoch, total_iter):
    return (epoch / total_epoch) * total_iter

def main():
    options = Options()
    options.initialize()
    modify_options(options)
    options.parse()

    print(options.pretty_str())
    run_cls = find_run_using_name(options.general.run)
    run = run_cls(options)

    train_loader = run.get_train_loader()
    num_iter = len(train_loader)

    general_opt = options.general
    cur_iter = 0

    run.setup()

    for epoch in range(1, general_opt.epoch+1):
        for i, data in enumerate(train_loader):
            cur_iter += 1
            run.iterate(data)

            if cur_iter % general_opt.print_iter == 0:
                float_epoch = cur_iter / num_iter
                run.log_and_visualize_iteration(epoch, cur_iter)
                print("training progress: {}/{}".format(float_epoch, general_opt.epoch))

        if epoch % general_opt.save_epoch == 0:
            run.save_checkpoint(epoch)
            print("checkpoint saved at {}th epoch".format(epoch))

        run.end_epoch()

if __name__ == '__main__':
    main()
