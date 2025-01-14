import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="5")
    parser.add_argument('--model', type=str, default='simplecnn', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--type', type=str, default='mode')
    parser.add_argument('--partition', type=str, help='the data partitioning strategy')
    parser.add_argument('--num_local_iterations', type=int, default=200, help='number of local iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')   
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')  
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, help="Data directory")
    parser.add_argument('--beta', type=float, default=0.1,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--skew_class', type=int, default = 2, help='The parameter for the noniid-skew for data partitioning')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer') 
    parser.add_argument('--concen_loss', type=str, default='uniform_norm', choices=['norm', 'uniform_norm'], help='How to measure the modle difference')
    parser.add_argument('--weight_norm', type=str, default='relu', choices=['sum', 'softmax', 'abs', 'relu', 'sigmoid'], help='How to measure the modle difference')
    parser.add_argument('--difference_measure', type=str, default='all', help='How to measure the model difference')
    
    parser.add_argument('--alpha1', type=float, default=0.5, help='Hyper-parameter to avoid concentration')
    parser.add_argument('--alpha2', type=float, default=2.5)
    parser.add_argument('--alpha3', type=float, default=1.0)
    parser.add_argument('--T', type=float, default=0.5, help='divided by the temperature')


    # attack
    parser.add_argument('--attack_type', type=str, default="inv_grad")
    parser.add_argument('--attack_ratio', type=float, default=0.0)
    
    args = parser.parse_args()
    if args.type=='mode':
        args.partition='noniid'
    elif args.type=='seve':
        args.partition='noniid-skew'
    elif args.type=='extr':
        args.partition='noniid-skew'
        args.n_parties=5
    elif args.type=='homo':
        args.partition='homo'
    cfg = dict()
    cfg["comm_round"] = args.comm_round
    cfg["optimizer"] = args.optimizer
    cfg["lr"] = args.lr
    cfg["epochs"] = args.epochs
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fashionMnist', 'yahoo_answers'}:
        cfg['classes_size'] = 10
    elif args.dataset == 'cifar100':
        cfg['classes_size'] = 100
    elif args.dataset == 'tinyimagenet':
        cfg['classes_size'] = 200
    cfg['client_num'] = args.n_parties
    cfg['model_name'] = args.model
    cfg['self_wight'] = 'loss'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.datadir=os.path.join('./data/',args.dataset)

    return args , cfg
