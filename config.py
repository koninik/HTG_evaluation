
#classes = '_!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '


#cdict = {c:i for i,c in enumerate(classes)}
#icdict = {i:c for i,c in enumerate(classes)}

k = 1
cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]

head_cfg = (256, 6)  # (hidden , num_layers) num_layers = 3 originally

#head_type = 'rnn'

flattening='maxpool'
#flattening='concat'

stn=False

max_epochs = 700

#batch_size = 300
#batch_size = 16
#level = "line"
#fixed_size = (4 * 32, 4 * 256)
#fixed_size = (1 * 64, 256)

batch_size = 100
level = "word"
fixed_size = (1 * 64, 256)

save_path = './saved_models/'
load_code = None