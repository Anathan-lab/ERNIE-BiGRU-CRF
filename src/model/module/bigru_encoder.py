import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiGRUEncoder(nn.Module):
    """
    BIGRU encoder.
    output the score of all labels.
    """

    def __init__(self, label_size: int, input_dim:int,
                 hidden_dim: int,
                 drop_gru:float=0.5,
                 num_gru_layers: int =1):
        super(BiGRUEncoder, self).__init__()

        self.label_size = label_size
        print("[Model Info] Input size to GRU: {}".format(input_dim))
        print("[Model Info] GRU Hidden Size: {}".format(hidden_dim))
        self.gru = nn.GRU(input_dim, hidden_dim // 2, num_layers=num_gru_layers, batch_first=True, bidirectional=True)
        self.drop_gru = nn.Dropout(drop_gru)
        self.hidden2tag = nn.Linear(hidden_dim, self.label_size)

    def forward(self, word_rep: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Encoding the input with BiGRU
        :param word_rep: (batch_size, sent_len, input rep size)
        :param word_seq_lens: (batch_size, 1)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """
        sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len.cpu(), True)
        gru_out, _ = self.gru(packed_words, None)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.
        feature_out = self.drop_gru(gru_out)

        outputs = self.hidden2tag(feature_out)
        return outputs[recover_idx]


