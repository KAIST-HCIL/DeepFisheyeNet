from option.options import Options
from preset import modify_options
from dataset import *
from run import find_run_using_name

def main():
    options = Options()
    options.initialize()
    modify_options(options)
    options.parse()

    print(options.pretty_str())

    run_cls = find_run_using_name(options.general.run)
    run = run_cls(options)

    test_loader = run.get_test_loader(shuffle = True)

    run.setup()

    for i, data in enumerate(test_loader):
        print("testing {}th data".format(i))
        run.test(data, i)

    run.end_test()

if __name__ == '__main__':
    main()
