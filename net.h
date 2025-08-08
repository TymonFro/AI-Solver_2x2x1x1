#include<bits/stdc++.h>
//#define int long long
// #define pr pair<int,int>
// #define f first 
// #define s second 
#define F(I,K) for(int I = 0; I<K;I++)
#define F1(I,K) for(int I = 1; I<=K;I++)
using namespace std;
const double lr = 0.005;
const double gam = 0.95;
const int in_layer = 24;
const int hidden_layers = 32;
const int out_layer = 7;
const int batch_size = 64;
const double maxNorm = 1.0;
double epsilon = 1.0;      
double epsilon_min = 0.1;    
double epsilon_decay = 0.995;

struct memory{
    vector<double> s;
    int a;
    double reward;
    vector<double> s_next;
    bool done;
};

vector<double> get_Q_values(const vector<double>& o_v) {
    vector<double> Q(out_layer-1);
    double mean = 0.0;
    F(a,out_layer-1){
        mean += o_v[a];
    } mean /= (out_layer-1);

    F(a,out_layer-1){
        Q[a] = o_v[out_layer-1] + (o_v[a] - mean);
    }
    return Q;
}

double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}
double dsigmoid(double x){
    return x * (1.0 - x);
}
double relu(double x){
    return max(0.0, x);
}
double drelu(double x){
    return x > 0 ? 1.0 : 0.0;
}
double elu(double x){
    return x > 0 ? x : exp(x) - 1.0;
}
double delu(double x){
    return x > 0 ? 1 : exp(x);
}