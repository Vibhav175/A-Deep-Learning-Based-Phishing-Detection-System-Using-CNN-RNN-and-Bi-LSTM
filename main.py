from preprocess_data import preprocess_data
from split_data import split_data
from cnn import build_cnn_model, train_cnn_model
from rnn import build_rnn_model, train_rnn_model
from bilstm import build_bidirectional_lstm_model, train_bidirectional_lstm_model
from evaluations import evaluate_model

X, y, max_len = preprocess_data('filtered_malicious_phish_frfrfr.csv', 'type', 5000)

X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=5)

cnn_model = build_cnn_model(5000, max_len)
rnn_model = build_rnn_model(5000, max_len)
bidirectional_lstm_model = build_bidirectional_lstm_model(tokenizer, max_len)


bilstm_history = train_bidirectional_lstm_model(bidirectional_lstm_model, X_train, y_train, X_test, y_test, epochs=5, batch_size=32)
rnn_history = train_rnn_model(rnn_model, X_train, y_train, X_test, y_test, epochs=5, batch_size=32)
cnn_history = train_cnn_model(cnn_model, X_train, y_train, X_test, y_test, epochs=5, batch_size=32)



evaluate_model(cnn_model, cnn_history, X_test, y_test, save_results_dir='results_cnn')
evaluate_model(bidirectional_lstm_model, bilstm_history, X_test, y_test, save_results_dir='results_bidirectional_lstm')
evaluate_model(rnn_model, rnn_history, X_test, y_test, save_results_dir='results_rnn')