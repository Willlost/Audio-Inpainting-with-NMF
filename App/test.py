from ABE_v2 import ABE
import random
import medleydb as mdb
import librosa

class test():
    def __init__(self):
        mtrack_gen = mdb.load_all_multitracks(['V2'])
        self.mix_path = [i.mix_path for i in mtrack_gen]
        self.random_sample = self.mix_path[random.randint(0,len(self.mix_path)-1)]
        print(self.random_sample)
        self.mix_path.remove(self.random_sample)
        self.test_data = librosa.load(self.random_sample,sr=2756.25,duration=10)
        self.train_data = [librosa.load(self.random_sample,sr=44100,duration=10),None,None]

    def _find_train_data(self):
        esr = 100
        for i in self.mix_path:
            Y = librosa.load(i,sr=44100,duration=10)
            abe = ABE(Y,self.train_data[0],self.train_data[0])
            _,spec_loss = abe.loss(mode=['esr'])
            if spec_loss['ESR'] < esr:
                self.train_data[1] = Y
                esr = spec_loss['ESR']

        for j in self.mix_path:
            Y = librosa.load(j,sr=44100,duration=10)
            abe = ABE(Y,self.train_data[0],self.train_data[0])
            _,spec_loss = abe.loss(mode=['esr'])
            if spec_loss['ESR'] > esr:
                self.train_data[2] = Y
                esr = spec_loss['ESR']

    def run(self,train_index):
        try:
            self._find_train_data()
            abe = ABE(self.test_data,self.train_data[train_index],self.train_data[0])
            _,sig,sr = abe.apply()
            abe.save('Inpainted.wav',sig,sr)
            abe.save('test_data.wav',self.test_data[0],self.test_data[1])
            abe.save('train_data.wav',self.train_data[0][0],self.train_data[0][1])
            w_loss,s_loss = abe.loss(mode=['l1','l2','esr'])
            print(f'{w_loss}\n{s_loss}')
            abe.specshow(save=True)
            print('Test successful. ')
        except Exception:
            raise Exception('Test failed. ')

t = test()
t.run(0)