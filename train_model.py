import argparse
from time import time
import pickle as pk
import torch.optim as optim
from model import RNS
from evaluation import evaluate_ranking
from interactions import Interactions
from utils import *
class Recommender(object):
    def __init__(self,
                 n_iter=None,
                 batch_size=None,
                 l2=None,
                 neg_samples=None,
                 test_neg=None,
                 learning_rate=None,
                 use_cuda=False,
                 model_args=None):
        self._num_items = None
        self._num_users = None
        self._net = None
        self.model_args = model_args
        self._batch_size = batch_size
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._neg_samples = neg_samples
        self._test_neg = test_neg
        self._device = torch.device("cuda" if use_cuda else "cpu")
        self.test_sequence = None
        self._candidate = dict()
    @property
    def _initialized(self):
        return self._net is not None
    def _initialize(self, interactions):
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users
        self.test_sequence = interactions.test_sequences
        f1 = open('data/reviews_Amazon_Instant_Video.json/u_text', 'rb')
        f2 = open('data/reviews_Amazon_Instant_Video.json/i_text', 'rb')
        f3 = open('data/reviews_Amazon_Instant_Video.json/vocabulary', 'rb')
        u = pk.load(f1)
        u_text = np.array([uu.flatten() for uu in u.values()])
        i = pk.load(f2)
        i_text = np.array([ii.flatten() for ii in i.values()])
        vocabulary = pk.load(f3)
        self._net = RNS(self._num_users,
                          self._num_items,
                          self.model_args, u_text, i_text, vocabulary).to(self._device)
        self._optimizer = optim.Adam(self._net.parameters(),
                                     weight_decay=self._l2,
                                     lr=self._learning_rate)
    def fit(self, train, test, verbose=False):
        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets
        users_np = train.sequences.user_ids.reshape(-1, 1)
        L, T = train.sequences.L, train.sequences.T
        n_train = sequences_np.shape[0]
        output_str = 'total training instances: %d' % n_train
        print(output_str)
        if not self._initialized:
            self._initialize(train)
        start_epoch = 0
        best_p1, best_p5, best_p10, best_r1, best_r5, best_r10, best_map, best_n5, best_h5, best_f5 \
            = [0 for _ in range(10)]
        for epoch_num in range(start_epoch, self._n_iter):
            t1 = time()
            self._net.train()
            users_np, sequences_np, targets_np = shuffle(users_np,
                                                         sequences_np,
                                                         targets_np)
            negatives_np = self._generate_negative_samples(users_np, train, n=self._neg_samples)
            users, sequences, targets, negatives = (torch.from_numpy(users_np).long(),
                                                    torch.from_numpy(sequences_np).long(),
                                                    torch.from_numpy(targets_np).long(),
                                                    torch.from_numpy(negatives_np).long())
            users, sequences, targets, negatives = (users.to(self._device),
                                                    sequences.to(self._device),
                                                    targets.to(self._device),
                                                    negatives.to(self._device))
            epoch_loss = 0.0
            for (minibatch_num,
                 (batch_users,
                  batch_sequences,
                  batch_targets,
                  batch_negatives)) in enumerate(minibatch(users,
                                                           sequences,
                                                           targets,
                                                           negatives,
                                                           batch_size=self._batch_size)):
                items_to_predict = torch.cat((batch_targets, batch_negatives), 1)
                items_prediction = self._net(batch_sequences,
                                             batch_users,
                                             items_to_predict)
                (targets_prediction,
                 negatives_prediction) = torch.split(items_prediction,
                                                     [batch_targets.size(1),
                                                      batch_negatives.size(1)], dim=1)
                self._optimizer.zero_grad()
                positive_loss = -torch.mean(
                    torch.log(torch.sigmoid(targets_prediction)))
                negative_loss = -torch.mean(
                    torch.log(1 - torch.sigmoid(negatives_prediction)))
                loss = positive_loss + negative_loss
                epoch_loss += loss.item()
                loss.backward()
                self._optimizer.step()
            epoch_loss /= minibatch_num + 1
            t2 = time()
            if verbose and (epoch_num + 1) % 1 == 0:
                precision, recall, mean_aps, ndcgs, hrs, f1s = evaluate_ranking(self, test, train, k=[1, 5, 10])
                output_str = "Epoch %d [%.1f s]\tloss=%.5f, map=%.5f, " \
                             "NDCG@5=%.5f, HR@5=%.5f, f1@5=%.5f, "\
                             "prec@5=%.5f, recall@5=%.5f, [%.1f s]" % (epoch_num + 1,
                                                                         t2 - t1,
                                                                         epoch_loss,
                                                                         mean_aps,
                                                                         np.mean(ndcgs[1]),
                                                                         np.mean(hrs[1]),
                                                                         np.mean(f1s[1]),
                                                                         np.mean(precision[1]),
                                                                         np.mean(recall[1]),
                                                                         time() - t2)
                print(output_str)
                best_p1 = np.mean(precision[0]) if np.mean(precision[0]) > best_p1 else best_p1
                best_p5 = np.mean(precision[1]) if np.mean(precision[1]) > best_p5 else best_p5
                best_p10 = np.mean(precision[2]) if np.mean(precision[2]) > best_p10 else best_p10
                best_r1 = np.mean(recall[0]) if np.mean(recall[0]) > best_r1 else best_r1
                best_r5 = np.mean(recall[1]) if np.mean(recall[1]) > best_r5 else best_r5
                best_r10 = np.mean(recall[2]) if np.mean(recall[2]) > best_r10 else best_r10
                best_map = mean_aps if mean_aps > best_map else best_map
                best_n5 = np.mean(ndcgs[1]) if np.mean(ndcgs[1]) > best_n5 else best_n5
                best_h5 = np.mean(hrs[1]) if np.mean(hrs[1]) > best_h5 else best_h5
                best_f5 = np.mean(f1s[1]) if np.mean(f1s[1]) > best_f5 else best_f5
            else:
                output_str = "Epoch %d [%.1f s]\tloss=%.5f [%.1f s]" % (epoch_num + 1,
                                                                        t2 - t1,
                                                                        epoch_loss,
                                                                        time() - t2)
                print(output_str)
        best_str = "best_p5=%.5f, best_r5=%.5f, best_f5=%.5f, best_n5=%.5f, best_h5=%.5f" \
                   % (best_p5, best_r5, best_f5, best_n5, best_h5)
        print(best_str)
    def _generate_negative_samples(self, users, interactions, n):
        users_ = users.squeeze()
        negative_samples = np.zeros((users_.shape[0], n), np.int64)
        if not self._candidate:
            all_items = np.arange(interactions.num_items)
            train = interactions.tocsr()
            for user, row in enumerate(train):
                self._candidate[user] = list(set(all_items) - set(row.indices))
        for i, u in enumerate(users_):
            for j in range(n):
                x = self._candidate[u]
                negative_samples[i, j] = x[
                    np.random.randint(len(x))]
        return negative_samples
    def predict(self, user_id, item_ids=None):
        if self.test_sequence is None:
            raise ValueError('Missing test sequences, cannot make predictions')
        self._net.eval()
        with torch.no_grad():
            sequences_np = self.test_sequence.sequences[user_id, :]
            sequences_np = np.atleast_2d(sequences_np)
            if item_ids is None:
                item_ids = np.arange(self._num_items).reshape(-1, 1)
            sequences = torch.from_numpy(sequences_np).long()
            item_ids = torch.from_numpy(item_ids).long()
            user_id = torch.from_numpy(np.array([[user_id]])).long()
            user, sequences, items = (user_id.to(self._device),
                                      sequences.to(self._device),
                                      item_ids.to(self._device))
            out = self._net(sequences,
                            user,
                            items,
                            for_pred=True)
        return out.cpu().numpy().flatten()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', type=str, default='data/reviews_Amazon_Instant_Video.json/video_train.csv')
    parser.add_argument('--test_root', type=str, default='data/reviews_Amazon_Instant_Video.json/video_test.csv')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--n_iter', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2018)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--test_neg', type=int, default=100)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    config = parser.parse_args()
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--drop', type=float, default=0.3)
    model_parser.add_argument('--ac_conv', type=str, default='relu')
    model_parser.add_argument('--ac_fc', type=str, default='relu')
    model_parser.add_argument('--dim', type=int, default=25, help='dimension of word embeddings')
    model_parser.add_argument('--nt', type=int, default=2, help='number of text cnn filters for each size')
    model_parser.add_argument('--nk', type=int, default=5, help='number of aspects')
    model_parser.add_argument('--alpha', type=float, default=0.1, help='weight of sequential preference')
    model_config = model_parser.parse_args()
    model_config.L = config.L
    set_seed(config.seed,
             cuda=config.use_cuda)
    train = Interactions(config.train_root)
    train.to_sequence(config.L, config.T)
    test = Interactions(config.test_root,
                        user_map=train.user_map,
                        item_map=train.item_map)
    model = Recommender(n_iter=config.n_iter,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate,
                        l2=config.l2,
                        neg_samples=config.neg_samples,
                        test_neg=config.test_neg,
                        model_args=model_config,
                        use_cuda=config.use_cuda)
    model.fit(train, test, verbose=True)
