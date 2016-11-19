#include <bits/stdc++.h>
#include <omp.h>
//#include "mpi.h"
using namespace std;

#define p(x) cout << #x << " = "<< x<< endl
#define min(a,b) a<b ? a : b
typedef vector<double> d1;
typedef vector<d1> d2;
typedef vector<d2> d3;
typedef vector<int> i1;

d3 xTrain, xTest;
d2 tTrain, tTest;

int argmax(d1 x){
  int maxIndex=0;
  double maxValue=x[0];
  for (int i=1; i<x.size(); i++){
    if (x[i] > maxValue){
      maxValue = x[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

void toRank2(d1&x, int rows, d2& y){
  for (int i=0; i<x.size()/rows; i++){
    y.emplace_back();
    for (int row=0; row<rows; row++){
      y[i].push_back(x[i*rows+row]);
    }
  }
}

void toRank3(d1& x, int rows, int cols, d3& y){
  for (int i=0; i<x.size()/rows/cols; i++){
    y.emplace_back();
    for (int row=0; row<rows; row++){
      y[i].emplace_back();
      for (int col=0; col<cols; col++){
        y[i][row].push_back(x[i*rows*cols+row*cols+col]);
      }
    }
  }
}

d2 zeros(int rows, int cols){
  return d2(rows, d1(cols, 0));
}

d3 zeros(int x, int rows, int cols){
  return d3(x, d2(rows, d1(cols, 0)));
}

void readFiles(int train=1000, int test=100){
  d1 _xTrain, _xTest;
  d1 _tTrain, _tTest;

  string line;
  int i=0;

  ifstream fXTrain("xTrain.csv");
  while(i++<train*pow(28,2) && getline(fXTrain, line) ){
    //cout << i << endl;
    _xTrain.push_back(stod(line));
  }
  i=0;
  toRank3(_xTrain, 28, 28, xTrain);

  ifstream fXTest("xTest.csv");
  while(i++<test*pow(28,2) && getline(fXTest, line)){
    _xTest.push_back(stod(line));
  }
  i=0;
  toRank3(_xTest, 28, 28, xTest);

  ifstream fTTrain("tTrain.csv");
  while(i++<train*10 && getline(fTTrain, line)){
    _tTrain.push_back(stoi(line));
  }
  i=0;
  toRank2(_tTrain, 10, tTrain);

  ifstream fTTest("tTest.csv");
  while(i++<test*10 && getline(fTTest, line)){
    _tTest.push_back(stoi(line));
  }
  i=0;
  toRank2(_tTest, 10, tTest);
}

d1 getRandomDoubles(int size, double mean=0, double standard_deviation=1){
  static normal_distribution<double> distribution(mean, standard_deviation);
  int seed=time(NULL);
  static default_random_engine generator(seed);
  d1 data(size);
  generate(data.begin(), data.end(), []() { return distribution(generator); });
  return data;
}

d2 getRandomDoubles(int rows, int cols, double mean=0, double standard_deviation=1){
  d1 d = getRandomDoubles(rows*cols, mean, standard_deviation);
  d2 e;
  toRank2(d, cols, e);
  return e;
}

d3 getRandomDoubles(int depth, int rows, int cols, double mean=0, double standard_deviation=1){
  d1 d = getRandomDoubles(depth*rows*cols, mean, standard_deviation);;
  d3 e;
  toRank3(d, rows, cols, e);
  return e;
}

void print(d1 x){
  for (double d: x)
  cout << d << endl;
  cout << endl;
}

void print(d2 x){
  for (auto row: x){
    for (double d: row){
      cout << d << " ";
    }
    cout << endl;
  }
  cout << endl;
}

void print(d3 x){
  for (d2 X: x)
  print(X);
}

struct Signal{
  d2 x, z1, a1, z2, a2;
  d1 z3, y;
};

struct Derivative{
  d3 wFull;
  d2 wConv, bConv, bPool;
  d1 bFull;
};

struct NeuralNet{
  int nInputs, nConv, nPool=2, nOutputs, pooledSize;
  d2 wConv, bConv, bPool;
  d3 wFull;
  d1 bFull;

  NeuralNet(int _nInputs, int _nConv, int _nOutputs){
    nInputs = _nInputs;
    nConv = _nConv;
    nOutputs = _nOutputs;
    pooledSize = nInputs/nPool;

    wConv = getRandomDoubles(nConv, nConv);
    bConv = getRandomDoubles(nInputs, nInputs);
    bPool = getRandomDoubles(pooledSize, pooledSize);
    wFull = getRandomDoubles(nOutputs, pooledSize, pooledSize);
    bFull = getRandomDoubles(nOutputs);
  }

  d1 activate(d1& x){
    d1 activated(x.size(), 0);
    for (int h=0; h<2; h++)
    for (int i=0; i<x.size(); i++)
    activated[i] = x[i]>0 ? x[i]: 0;//max(x[i], 0.);
    return activated;
  }

  d2 activate(d2& x){
    d2 activated(x.size(), d1(x[0].size(), 0));
    for (int h=0; h<2; h++)
    for (int i=0; i<x.size(); i++)
    for (int j=0; j<x.size(); j++)
    activated[i][j] = x[i][j]>0 ? x[i][j] : 0;
    return activated;
  }

  d2 maxPool(d2& x){
    d2 pooled(x.size()/nPool, d1(x[0].size()/nPool, 0));

    for (int i=0; i<x.size(); i+=2)
    for (int j=0; j<x[0].size(); j+=2){
      pooled[i/2][j/2] = max(max(x[i][j], x[i+1][j]), max(x[i+1][j], x[i+1][j+1]));
    }
    return pooled;
  }

  d2 maxPoolDerivative(d2& x){
    d2 derivative(x.size(), d1(x[0].size(), 0));
    d2 m = maxPool(x);
    for (int i=0; i<x.size(); i++)
    for (int j=0; j<x[0].size(); j++){
      derivative[i][j] = m[i/2][j/2] == x[i][j];
    }
    return derivative;
  }

  d2 convolve(d2& w, d2& x){
    d2 convolved(x.size(), d1(x[0].size(), 0));
    int wCenterX = w[0].size() / 2;
    int wCenterY = w.size() / 2;
    int rows = x.size(), cols = x[0].size();
    int wRows = w.size(), wCols = w[0].size();

    #pragma omp parallel for
    for(int i=0; i < rows; i++)
    for(int j=0; j < cols; j++)
    for(int m=0; m < w.size(); m++){
      int mm = w.size() - 1 - m;
      for(int n=0; n < wCols; n++){
        int nn = wCols - 1 - n;
        int ii = i + (m - wCenterY);
        int jj = j + (n - wCenterX);
        if (ii >= 0 && ii < rows && jj >= 0 && jj < cols)
        convolved[i][j] += x[ii][jj] * w[mm][nn];
      }
    }
    return convolved;
  }

  d1 add(d1 x, d1 y){
    d1 sum(x.size(), 0);
    for (int i=0; i<x.size(); i++)
    sum[i] = x[i] + y[i];
    return sum;
  }

  d2 add(d2 x, d2 y){
    d2 sum(x.size(), d1(x[0].size(), 0));
    for (int i=0; i<x.size(); i++)
    for (int j=0; j<x[0].size(); j++)
    sum[i][j] = x[i][j] + y[i][j];
    return sum;
  }

  d3 add(d3 x, d3 y){
    d3 sum(x.size(), d2(x[0].size(), d1(x[0][0].size(), 0)));
    for (int i=0; i<x.size(); i++)
    for (int j=0; j<x[0].size(); j++)
    for (int k=0; k<x[0][0].size(); k++)
    sum[i][j][k] = x[i][j][k] + y[i][j][k];
    return sum;
  }

  d3 multiply(d3 x, double d){
    for (int i=0; i<x.size(); i++)
    for (int j=0; j<x[0].size(); j++)
    for (int k=0; k<x[0][0].size(); k++)
    x[i][j][k] *= d;
    return x;
  }

  d1 fullyConnect(d3 w, d2 x){
    d1 connected(w.size(), 0);
    for (int i=0; i<w.size(); i++)
    for (int j=0; j<w[0].size(); j++)
    for (int k=0; k<w[0][0].size(); k++)
    connected[i] += w[i][j][k] * x[j][k];
    return connected;
  }

  d1 softMax(d1& x){
    d1 softMaxed(x.size(), 0);
    double largest = x[argmax(x)];
    double lndenom = largest;
    double expsum = 0;
    for (int i=0; i<x.size(); i++)
    expsum += exp(x[i]-largest);
    for (int i=0; i<x.size(); i++)
    softMaxed[i] = exp(x[i]-largest) / expsum;
    return softMaxed;
  }

  Signal forwardPropagate(d2& x){
    Signal s;
    s.x = x;
    s.z1 = add(convolve(wConv, x), bConv);
    s.a1 = activate(s.z1);
    s.z2 = add(maxPool(s.a1), bPool);
    s.a2 = activate(s.z2);
    s.z3 = add(fullyConnect(wFull, s.a2), bFull);
    s.y = softMax(s.z3);
    return s;
  }



  d3 wFullDerivative(Signal& s, d1& t){
    d3 out = zeros(nOutputs, pooledSize, pooledSize);

    for (int a=0; a<nOutputs; a++)
    for (int f=0; f<pooledSize; f++)
    for (int g=0; g<pooledSize; g++){
      out[a][f][g] = s.a2[f][g]*(s.y[a]-t[a]);
    }
    return out;
  }

  d2 wConvDerivative(Signal&s, d1& t){
    d2 out = zeros(wConv.size(), wConv[0].size());
    d2 mPD = maxPoolDerivative(s.a1);
    int dmax = s.z1.size(), emax=s.z1[0].size();
    for (int l=0; l<wConv.size(); l++)
    for (int m=0; m<wConv[0].size(); m++){
      if (s.z1[l][m] > 0)
      for (int h=0; h<t.size(); h++)
      for (int i=0; i<s.y.size(); i++)
      {
        double hi_pre = t[h] * (s.y[i] - i==h);
        for (int f=0; f<2; f++)
        for (int g=0; g<2; g++){
          double outlm=0;
          int F = min(l/2+f, s.z2.size()-1);
          int G = min(m/2+g, s.z2[0].size()-1);
          double prefactor = wFull[i][F][G] *  (s.z2[F][G]>0)*mPD[F][G];
          for (int d=l; d<dmax; d++)//these upper bounds are suspect...
          for (int e=m; e<emax; e++){
            if (s.z1[d][e]>0)
            outlm += s.x[d-l][e-m];
          }
          out[l][m] = outlm*prefactor*hi_pre;
        }
      }
    }
    return out;
  }

  //
  d1 bFullDerivative(Signal& s, d1& t){
    d1 out(t.size());
    for (int i=0; i<t.size(); i++)
    out[i] = s.y[i] - t[i];
    return out;
  }

  d2 bPoolDerivative(Signal& s, d1& t){
    d2 out = zeros(bPool.size(), bPool[0].size());
    for (int l=0; l<out.size(); l++)
    for (int m=0; m<out[0].size(); m++)
    for (int i=0; i<s.y.size(); i++)
    for (int h=0; h<t.size(); h++)
    out[l][m] -= t[h]*s.y[h]*(i==h - s.y[i])*wFull[i][l][m]*(s.z2[l][m] > 0);
    return out;
  }

  d2 bConvDerivative(Signal&s, d1& t){
    d2 out = zeros(bConv.size(), bConv[0].size());
    for (int l=0; l<bConv.size(); l++)
    for (int m=0; m<bConv[0].size(); m++){
      d2 mPD = maxPoolDerivative(s.a1);
      for (int h=0; h<t.size(); h++)
      for (int i=0; i<s.y.size(); i++)
      for (int f=0; f<2; f++)
      for (int g=0; g<2; g++){
        int F = min(l/2+f, s.z2.size()-1);
        int G = min(m/2+g, s.z2[0].size()-1);
        out[l][m] -= t[h]*(i==h - s.y[i])*
        wFull[i][F][G]*(s.z2[F][G]>0)*mPD[F][G]*(s.z1[l][m] > 0);
      }
    }
    return out;
  }

  Derivative getDerivatives(Signal& s, d1& t){
    Derivative d;
    d.wConv = wConvDerivative(s, t);
    d.wFull = wFullDerivative(s, t);
    d.bConv = bConvDerivative(s, t);
    d.bPool = bPoolDerivative(s, t);
    d.bFull = bFullDerivative(s, t);
    return d;
  }

  void learn(d2& x, d1& t, double learning_rate){

    Signal s = forwardPropagate(x);
    Derivative d = getDerivatives(s, t);

    for (int l=0; l<wConv.size(); l++)
    for (int m=0; m<wConv[0].size(); m++){
      wConv[l][m] -= d.wConv[l][m]*learning_rate;
    }

    for (int a=0; a<wFull.size(); a++)
    for (int f=0; f<wFull[0].size(); f++)
    for (int g=0; g<wFull[0][0].size(); g++){
      wFull[a][f][g] -= d.wFull[a][f][g]*learning_rate;
    }
    //

    for (int l=0; l<bConv.size(); l++)
    for (int m=0; m<bConv[0].size(); m++){
      bConv[l][m] -= d.bConv[l][m]*learning_rate;
    }

    for (int l=0; l<bPool.size(); l++)
    for (int m=0; m<bPool[0].size(); m++){
      bPool[l][m] -= d.bPool[l][m]*learning_rate;
    }

    for (int j=0; j<bFull.size(); j++){
      bFull[j] -= d.bFull[j]*learning_rate;
    }
  }

  double error(d3 X, d2 T){
    double e = 0;
    for (int i=0; i<X.size(); i++){
      Signal s = forwardPropagate(X[i]);
      for (int j=0; j<s.y.size(); j++){
        if (s.y[j] && T[i][j])
        e += log(s.y[j]);
        // else
        // e += -30*T[i][j];
      }
    }
    return -e/X.size();
  }

  double percentCorrect(d3 X, d2 T){
    double correct=0, incorrect=0;
    for (int i=0; i<X.size(); i++){
      Signal s = forwardPropagate(X[i]);
      if (argmax(s.y) == argmax(T[i])){
        correct++;
      }
      else{
        incorrect++;
      }
    }
    return correct/(correct+incorrect);
  }

  d1 percentCorrectHist(d3 X, d2 T){
    d1 correct(T[0].size());
    d1 incorrect(T[0].size());
    for (int i=0; i<X.size(); i++){
      Signal s = forwardPropagate(X[i]);

      if (argmax(s.y) == argmax(T[i]))
      correct[argmax(T[i])] += 1.0;
      else
      incorrect[argmax(T[i])] += 1.0;
    }
    d1 percent(T[0].size());
    for (int i=0; i<T[0].size(); i++)
    if (correct[i] + incorrect[i])
    percent[i] = (double)correct[i]/(double)(correct[i]+incorrect[i]);
    return percent;
  }
};

int main(){
  readFiles(1000, 1000);
  NeuralNet skynet(28, 16, 10);

  int epochs = 100000;
  i1 order(xTest.size(), 0);
  iota(order.begin(), order.end(), 0);
  random_shuffle(order.begin(), order.end());
  double learning_rate = 1e-2;
  double last_error = 0;
  for (int i=0; i<epochs+1; i++){
    int j = order[i%order.size()];

    int print_skip = 100;
    d1 stats;
    if (i%print_skip == 0){
      stats = {(double)i,
        skynet.percentCorrect(xTrain, tTrain), skynet.percentCorrect(xTest, tTest),
        skynet.error(xTrain, tTrain), skynet.error(xTest, tTest),
        learning_rate};
        for (int i=0; i<stats.size(); i++)
        cout << stats[i] << " ";
        cout << endl;
      }
      skynet.learn(xTrain[j], tTrain[j], learning_rate);
      if (i==1000){
        learning_rate /= 5;
      }

    }
  }
