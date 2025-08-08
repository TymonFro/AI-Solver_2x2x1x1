#include<bits/stdc++.h>
#include"2x2x1x1.h"
#include"net.h"
//#define int long long
// #define pr pair<int,int>
// #define f first 
// #define s second 
#define F(I,K) for(int I = 0; I<K;I++)
#define F1(I,K) for(int I = 1; I<=K;I++)
using namespace std;
int scramble_moves = 5;

vector<vector<vector<double>>>h_w = {vector<vector<double>>(hidden_layers, vector<double>(in_layer)), vector<vector<double>>(hidden_layers, vector<double>(hidden_layers))};
vector<vector<double>>h_b(2, vector<double>(hidden_layers));
vector<vector<double>>o_w(out_layer, vector<double>(hidden_layers));
vector<double>o_b(out_layer);

vector<vector<vector<double>>>t_h_w = {vector<vector<double>>(hidden_layers, vector<double>(in_layer)), vector<vector<double>>(hidden_layers, vector<double>(hidden_layers))};
vector<vector<double>>t_h_b(2, vector<double>(hidden_layers));
vector<vector<double>>t_o_w(out_layer, vector<double>(hidden_layers));
vector<double>t_o_b(out_layer);

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

void load_net(){
    ifstream fin("2x2x1x1_net.txt");
    F(i,2){
        F(j,h_w[i].size()){
            F(k,h_w[i][j].size()){
                fin >> h_w[i][j][k];
            }
        }
    }
    F(i,2){
        F(j,h_b[i].size()){
            fin >> h_b[i][j];
        }
    }
    F(i,o_w.size()){
        F(j,o_w[i].size()){
            fin >> o_w[i][j];
        }
    }
    F(i,o_b.size()){
        fin >> o_b[i];
    }
    fin.close();
}

void save_net(){
    ofstream fout("2x2x1x1_net.txt");
    F(i,2){
        F(j,h_w[i].size()){
            F(k,h_w[i][j].size()){
                fout << h_w[i][j][k] << " ";
            }
            fout << endl;
        }
    }
    F(i,2){
        F(j,h_b[i].size()){
            fout << h_b[i][j] << " ";
        }
        fout << endl;
    }
    F(i,o_w.size()){
        F(j,o_w[i].size()){
            fout << o_w[i][j] << " ";
        }
        fout << endl;
    }
    F(i,o_b.size()){
        fout << o_b[i] << " ";
    }
    fout.close();
}

void gen_net(){
    F(i,2){
        F(j,h_w[i].size()){
            F(k,h_w[i][j].size()){
                h_w[i][j][k] = uniform_real_distribution<double>(-0.1, 0.1)(rng);
            }
        }
    }
    F(i,2){
        F(j,h_b[i].size()){
            h_b[i][j] = uniform_real_distribution<double>(-0.1, 0.1)(rng);
        }
    }
    F(i,o_w.size()){
        F(j,o_w[i].size()){
            o_w[i][j] = uniform_real_distribution<double>(-0.1, 0.1)(rng);
        }
    }
    F(i,o_b.size()){
        o_b[i] = uniform_real_distribution<double>(-0.1, 0.1)(rng);
    }
}

void copy_net(){
    F(i,2){
        F(j,h_w[i].size()){
            F(k,h_w[i][j].size()){
                t_h_w[i][j][k] = h_w[i][j][k];
            }
        }
    }
    F(i,2){
        F(j,h_b[i].size()){
            t_h_b[i][j] = h_b[i][j];
        }
    }
    F(i,o_w.size()){
        F(j,o_w[i].size()){
            t_o_w[i][j] = o_w[i][j];
        }
    }
    F(i,o_b.size()){
        t_o_b[i] = o_b[i];
    }
}

void forwardpropagation(vector<double>& in_v, vector<vector<double>>& h_v, vector<double>&o_v){
    F(l,h_w.size()){
        F(i, h_w[l].size()){
            h_v[l][i] = h_b[l][i];
            if(l == 0){
                F(j, in_v.size()){
                    h_v[l][i] += in_v[j] * h_w[l][i][j];
                }
            }else{
                F(j, h_w[l][i].size()){
                    h_v[l][i] += h_v[l-1][j] * h_w[l][i][j];
                }
            }
            h_v[l][i] = relu(h_v[l][i]);
        }
    }
    int ll = h_w.size()-1;
    int ll_size = h_w[ll].size();

    F(i,o_w.size()){
        o_v[i] = o_b[i];
        F(j,ll_size){
            o_v[i] += h_v[ll][j] * o_w[i][j];
        }
        //o_v[i] = sigmoid(o_v[i]);
    }
}

void forwardtargetpropagation(vector<double>& in_v, vector<double>&o_v){
    vector<vector<double>>h_v(2, vector<double>(hidden_layers, 0.0));
    F(l,t_h_w.size()){
        F(i, t_h_w[l].size()){
            h_v[l][i] = t_h_b[l][i];
            if(l == 0){
                F(j, in_v.size()){
                    h_v[l][i] += in_v[j] * t_h_w[l][i][j];
                }
            }else{
                F(j, t_h_w[l][i].size()){
                    h_v[l][i] += h_v[l-1][j] * t_h_w[l][i][j];
                }
            }
            h_v[l][i] = relu(h_v[l][i]);
        }
    }
    int ll = t_h_w.size()-1;
    int ll_size = t_h_w[ll].size();

    F(i,t_o_w.size()){
        o_v[i] = t_o_b[i];
        F(j,ll_size){
            o_v[i] += h_v[ll][j] * t_o_w[i][j];
        }
        //o_v[i] = sigmoid(o_v[i]);
    }
}

float computeL2Norm(const std::vector<float>& gradients) {
    float norm = 0.0f;
    for (float g : gradients) {
        norm += g * g;
    }
    return sqrt(norm);
}

// Function to clip gradients by a maximum norm
void clipGradients(std::vector<float>& gradients, float maxNorm) {
    float currentNorm = computeL2Norm(gradients);
    if (currentNorm > maxNorm) {
        float scale = maxNorm / (currentNorm + 1e-6f); // Add epsilon to prevent div-by-zero
        for (float& g : gradients) {
            g *= scale;
        }
    }
}

void backpropagate(vector<double>& Qs, vector<double>& in_v, vector<vector<double>>&h_v, vector<double>& o_v){
    vector<double> o_d(o_w.size(), 0.0);
    vector<vector<double>> h_d(2);
    h_d[0].resize(h_w[0].size(), 0.0);
    h_d[1].resize(h_w[1].size(), 0.0);

    o_d = Qs;

    int ll = h_w.size() - 1;
    for(int l = ll; l>=0; l--){
        F(i,h_w[l].size()){
            if(l == ll){
                F(j,o_w.size()){
                    h_d[l][i] += o_d[j] * o_w[j][i];
                }
                //h_d[l][i] *= delu(h_v[l][i]);
            }else{
                F(j,h_w[l+1].size()){
                    h_d[l][i] += h_d[l+1][j] * h_w[l+1][j][i];
                }
                //h_d[l][i] *= delu(h_v[l][i]);
                h_d[l][i] = h_d[l][i] * drelu(h_v[l][i]);
            }
        }
    }

     // clip
    double norm = 0.0;
    for(double cur : o_d) {
        norm += cur * cur;
    }
    for(int l = 0; l < h_w.size(); l++) {
        for(double cur : h_d[l]) {
            norm += cur * cur;
        }
    }
    norm = sqrt(norm);
    if (norm > maxNorm) {
        float scale = maxNorm / (norm + 1e-6f);
        for(double& cur : o_d) {
            cur *= scale;
        }
        for(int l = 0; l < h_w.size(); l++) {
            for(double& cur : h_d[l]) {
                cur *= scale;
            }
        }
    }

    F(i, o_w.size()){
        o_b[i] -= lr * o_d[i];
        F(j, h_w[ll].size()){ 
            o_w[i][j] -= lr * o_d[i] * h_v[ll][j];
        }
    }

    for(int l = ll; l>=0; l--){
        F(i,h_w[l].size()){
            if(l == 0){
                h_b[l][i] -= lr * h_d[l][i];
                F(j,in_v.size()){
                    h_w[l][i][j] -= lr * h_d[l][i] * in_v[j];
                }
            }else{
                h_b[l][i] -= lr * h_d[l][i];
                F(j,h_w[l][i].size()){
                    h_w[l][i][j] -= lr * h_d[l][i] * h_v[l-1][j];
                }
            }
        }
    }
}

signed main(){
    //ios_base::sync_with_stdio(true);
    //cin.tie(NULL);
    // wczytaj siec
    ifstream fin("2x2x1x1_net.txt");

    if(fin.good()){ //
        fin.close();
        load_net();
        copy_net();
    }else{
        gen_net();
        copy_net();
    }

    int il;
    cout<<"Enter number of epochs: ";
    cin>>il;

    deque<memory>buffer;

    for(int epoch = 0; epoch <  il; epoch++){
        epsilon = max(epsilon * epsilon_decay, epsilon_min);
        
        Cube2x2x1x1 cube = Cube2x2x1x1();

        int scramble;
        if(epoch < 1200){
            scramble = 1;
        }
        else if(epoch < 3000){
            scramble = uniform_int_distribution<int>(1, scramble_moves)(rng);
            scramble = min(scramble, 2);
        }
        else if(epoch < 10000){
            scramble = uniform_int_distribution<int>(1, scramble_moves)(rng);
            scramble = min(scramble, 3);
        }
        else if(epoch < 20000){
            scramble = uniform_int_distribution<int>(1, scramble_moves + 1)(rng);
            scramble = min(scramble, 6);
        }

        if(epoch == 3000){
            epsilon = 0.5;
        }
        if(epoch == 5000){
            epsilon = 1.0;
        }
        if(epoch == 7000){
            epsilon = 0.5;
        }
        // if(epoch == 15000){
        //     epsilon = 0.5;
        // }
        
        F(i,scramble){
            int move = uniform_int_distribution<int>(0, 5)(rng);
            cube = applyMove(cube, move);
        }

        for(int move_count = 0; move_count < scramble; move_count++){
            vector<double> in_v = encodeCube(cube);

            vector<vector<double>>h_v(2, vector<double>(hidden_layers, 0.0));
            vector<double>o_v(o_w.size(), 0.0);
            forwardpropagation(in_v, h_v, o_v);
            o_v = get_Q_values(o_v);

            int move;
            double rv = uniform_real_distribution<double>(0.0, 1.0)(rng);
            if(rv < epsilon){
                move = uniform_int_distribution<int>(0, 5)(rng);
            }else{
                move = max_element(o_v.begin(), o_v.end()) - o_v.begin();
            }

            Cube2x2x1x1 new_cube = applyMove(cube, move);
            vector<double> next_in_v = encodeCube(new_cube);

            if(isSolved(new_cube)){
                cout<<"Solved in "<<move_count+1<<" moves"<<endl;
                buffer.push_back({in_v, move, 1.0*scramble/((double)(move_count+1)), next_in_v, true});
                break;
            }else{
                //cout<<"move "<<move_count+1<<": "<<move<<endl;
                buffer.push_back({in_v, move, -0.1, next_in_v, false});
                cube = applyMove(cube, move);
                if(move_count == scramble-1){
                    cout<<"Unsolved from "<<scramble<<" moves"<<endl;
                }
            }

            if(buffer.size() >= batch_size){
                F(k,batch_size){
                    vector<vector<double>>h_v(2, vector<double>(hidden_layers, 0.0));

                    int idx = uniform_int_distribution<int>(0, buffer.size() - 1)(rng);
                    memory m = buffer[idx];

                    vector<double>o_v(o_w.size(), 0.0);
                    forwardpropagation(m.s, h_v, o_v);
                    o_v = get_Q_values(o_v);
                    vector<double> t_o_v(o_w.size(), 0.0);
                    forwardtargetpropagation(m.s_next, t_o_v);
                    t_o_v = get_Q_values(t_o_v);

                    double max_qsi = *max_element(t_o_v.begin(), t_o_v.end());

                    double y = m.reward + (m.done ? 0 : gam * max_qsi);
                    double error = o_v[m.a] - y;
                    
                    vector<double> target_Q (o_w.size(), 0.0);

                    target_Q[out_layer-1] = 2 * error;
                    F(i,out_layer-1){
                        if(i == m.a){
                            target_Q[i] = 2* error * (1 - (1/(double)(out_layer-1)));
                        }else{
                            target_Q[i] = -2 * error * (1 / (double)(out_layer-1));
                        }
                    }

                    backpropagate(target_Q, m.s, h_v, o_v);
                }
            }
        }

        while(buffer.size() > 10000){
            buffer.pop_front();
        }
        if(epoch % 100 == 0) {
            save_net();
            copy_net();
        }
    }

    //cout<<epsilon<<endl;

    save_net();
}