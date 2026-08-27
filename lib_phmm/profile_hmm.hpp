#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <string>
#include <utility>
#include <algorithm>
#include <optional>

using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

enum FlankState {
  N=0,
  B=1,
  E=2,
  C=3,
  J=4
};

const int MATCH_STATE = J + 1;


template<typename T>
void stream_vector(ostream& os, const vector<T>& vec) {
  for (int i = 0; i < (int)vec.size(); i++) {
    os << vec[i] << ",";
  }
}


class FlankTransitions {
public:
  FlankTransitions(float nn, float nb,
                   float ec, float ej,
                   float jj, float jb,
                   float cc,
                   vector<vector<float>> b_to_hmm,
                   vector<vector<float>> mm) {
    this->nn = log2(nn);
    this->nb = log2(nb);
    this->ec = log2(ec);
    this->ej = log2(ej);
    this->jj = log2(jj);
    this->jb = log2(jb);
    this->cc = log2(cc);
    for (int i = 0; i < (int)b_to_hmm.size(); i++) {
      vector<float> logs;
      for (int j = 0; j < (int)b_to_hmm[i].size(); j++) {
        logs.push_back(log2(b_to_hmm[i][j]));
      }
      this->b_to_hmm.push_back(logs);
    }

    for (int i = 0; i < (int)mm.size(); i++) {
      vector<float> logs;
      for (int j = 0; j < (int)mm[i].size(); j++) {
        logs.push_back(log2(mm[i][j]));
      }
      this->mm.push_back(logs);
    }
  }

  friend ostream& operator<<(ostream& os, const FlankTransitions& trans) {
    os << "  NN: " << trans.nn << " NB: " << trans.nb << "\n";
    os << "  EC: " << trans.ec << " EJ: " << trans.ej << "\n";
    os << "  JJ: " << trans.jj << " JB: " << trans.jb << "\n";
    os << "  CC: " << trans.cc << "\n";
    for (int i = 0; i < (int)trans.b_to_hmm.size(); i++) {
      os << "  BMi[" << i << "]: ";
      stream_vector(os, trans.b_to_hmm[i]);
      os << "\n";
    }
    for (int i = 0; i < (int)trans.mm.size(); i++) {
      os << "  MM[" << i << "]: ";
      stream_vector(os, trans.mm[i]);
      os << "\n";
    }
    return os;
  }

  // see Figure 1b in [1]
  float nn, nb;
  float ec, ej;
  float jj, jb;
  float cc;
  vector<vector<float>> b_to_hmm;  // [k][i] entry log probs
  vector<vector<float>> mm;        // [n][k] match self-transition log probs
};


inline Mat zeros(const int n, const int m) {
  return Mat(n, Vec(m, -INFINITY));
}


class Gaussian {
public:
  Gaussian(Vec& mean, Vec& variance) : mean(mean), variance(variance) {
    assert(mean.size() == variance.size());
    double log_std = 0;
    for (int i = 0; i < (int)variance.size(); i++) {
      log_std += log(sqrt(variance[i]));
    }
    scaler = (-(double)mean.size() / 2.0) * log(2.0 * M_PI);
    scaler = scaler - log_std;
  }

  friend ostream& operator<<(ostream& os, const Gaussian& gaussian) {
    os << "  mean: ";
    stream_vector(os, gaussian.mean);
    os << "\n  variance: ";
    stream_vector(os, gaussian.variance);
    return os;
  }

  double ll(const Vec& x) {
    assert(mean.size() == x.size());
    double error = 0;
    for (int i = 0; i < (int)x.size(); i++) {
      double e = (x[i] - mean[i]);
      error += (e * e) / variance[i];
    }
    return scaler - 0.5 * error;
  }

  const Vec& get_mean() const { return mean; }
  const Vec& get_variance() const { return variance; }

private:
  Vec mean;
  Vec variance;
  double scaler;
};


double log_add(double lx, double ly) {
  if (isinf(lx)) {
    return ly;
  }
  if (isinf(ly)) {
    return lx;
  }
  
  if (lx > ly) {
    return lx + log1p(exp(ly - lx));
  } else {
    return ly + log1p(exp(lx - ly));
  }
}


class MixtureModel {
public:
  MixtureModel(vector<Gaussian> components, Vec log_weights):
    components(components), log_weights(log_weights) {}

  double ll(const Vec &x) {
    double ll = -INFINITY;
    int n = log_weights.size();
    for(int i = 0; i < n; i++) {
      ll = log_add(ll, components[i].ll(x) + log_weights[i]);
    }
    return ll;
  }

  const vector<Gaussian>& get_components() const { return components; }
  const Vec& get_log_weights() const { return log_weights; }

private:
  vector<Gaussian> components;
  Vec log_weights;
};


class ProfileHMM {
public:
  ProfileHMM(vector<vector<Gaussian>>& pdf, FlankTransitions& trans)
      : pdf(pdf), trans(trans) {}

  friend ostream& operator<<(ostream& os, const ProfileHMM& hmm) {
    os << "================================\n";
    os << "Flank States:\n";
    os << "  NN: " << hmm.trans.nn << " NB: " << hmm.trans.nb << "\n";
    os << "  EC: " << hmm.trans.ec << " EJ: " << hmm.trans.ej << "\n";
    os << "  JJ: " << hmm.trans.jj << " JB: " << hmm.trans.jb << "\n";
    os << "  CC: " << hmm.trans.cc << "\n";
    for (int i = 0; i < (int)hmm.pdf.size(); i++) {
      os << "================================\n";
      os << "HMM [" << i << "]\n";
      os << "  BMi: ";
      stream_vector(os, hmm.trans.b_to_hmm[i]);
      os << "\n--------------------------------\n";
      for (int j = 0; j < (int)hmm.pdf[i].size(); j++) {
        os << "State M[" << j << "]\n" << hmm.pdf[i][j] << "\n";
      }
    }
    os << "================================\n";
    return os;
  }

  vector<vector<Gaussian>> pdf;
  FlankTransitions trans;
};


inline int match_state(int i, int hmm_idx, int n_match_states) {
  return hmm_idx * n_match_states + i + MATCH_STATE;
}


struct Pred {
  int time;
  int state;
};

inline string state_name(int state, int n_models, int match_state_per_model) {
  if (state == N) return "N";
  if (state == B) return "B";
  if (state == E) return "E";
  if (state == C) return "C";
  if (state == J) return "J";
  int idx = state - MATCH_STATE;
  int n   = idx / match_state_per_model;
  int k   = idx % match_state_per_model;
  return "M[" + to_string(n) + "][" + to_string(k) + "]";
}


inline pair<double, vector<Pred>> viterbi(const Mat& sequence, ProfileHMM& phmm,
                                           optional<MixtureModel> noise_pdf = nullopt) {
  int n_models = phmm.pdf.size();
  int match_state_per_model = phmm.pdf[0].size();
  int n_states = match_state_per_model * n_models + MATCH_STATE;
  int length = sequence.size();

  Mat W = zeros(length, n_states);
  vector<vector<Pred>> TB(length, vector<Pred>(n_states, {-1, -1}));

  W[0][N] = 0.0;
  W[0][B] = phmm.trans.nb;
  TB[0][N] = {0, N};
  TB[0][B] = {0, N};

  for (int i = 1; i < length; i++) {
    W[i][E] = -INFINITY;
    for (int n = 0; n < n_models; n++) {
      for (int k = 1; k < match_state_per_model; k++) {
        int cur  = match_state(k, n, match_state_per_model);
        int prev = match_state(k - 1, n, match_state_per_model);

        double emission   = phmm.pdf[n][k].ll(sequence[i]) / log(2.0);
        double from_prev  = W[i-1][prev];
        double from_self  = W[i-1][cur] + phmm.trans.mm[n][k]; // State-specific self-transition
        double from_entry = W[i-1][B] + phmm.trans.b_to_hmm[n][k];

        double best_val = from_prev;
        int best_prev_state = prev;

        if (from_self > best_val) {
          best_val = from_self;
          best_prev_state = cur;
        }
        if (from_entry > best_val) {
          best_val = from_entry;
          best_prev_state = B;
        }

        W[i][cur]  = emission + best_val;
        TB[i][cur] = {i-1, best_prev_state};

        if (W[i][cur] > W[i][E]) {
          W[i][E]  = W[i][cur];
          TB[i][E] = {i, cur};
        }
      }
    }

    // Silent by default (noise_emission=0), matching every existing
    // caller (train/processing.py never pass noise_pdf) exactly as
    // before: N/J/C carry no data-dependent signal, only transition
    // cost, so a frame only has to be *cheaper than the transition
    // cost* to get pulled into a match run -- there's no competing
    // hypothesis saying "this looks like background, not motif". When
    // find's --noise-wav supplies a noise_pdf, N/J/C become a real
    // rival hypothesis scored against the same data match states are,
    // same way ll()/log(2.0) is already used for match emissions.
    double noise_emission = noise_pdf.has_value() ? (noise_pdf->ll(sequence[i]) / log(2.0)) : 0.0;

    W[i][N]  = W[i-1][N] + phmm.trans.nn + noise_emission;
    TB[i][N] = {i-1, N};

    double from_jj = W[i-1][J] + phmm.trans.jj;
    double from_ej = W[i-1][E] + phmm.trans.ej;
    if (from_jj >= from_ej) { W[i][J] = from_jj + noise_emission; TB[i][J] = {i-1, J}; }
    else                     { W[i][J] = from_ej + noise_emission; TB[i][J] = {i-1, E}; }

    double from_cc = W[i-1][C] + phmm.trans.cc;
    double from_ec = W[i][E]   + phmm.trans.ec;
    if (from_cc >= from_ec) { W[i][C] = from_cc + noise_emission; TB[i][C] = {i-1, C}; }
    else                    { W[i][C] = from_ec + noise_emission;  TB[i][C] = {i,   E}; }

    double from_nb = W[i-1][N] + phmm.trans.nb;
    double from_jb = W[i][J]   + phmm.trans.jb;
    if (from_nb >= from_jb) { W[i][B] = from_nb; TB[i][B] = {i-1, N}; }
    else                    { W[i][B] = from_jb;  TB[i][B] = {i,   J}; }
  }

  vector<Pred> path;
  Pred cur = {length - 1, C};
  while (true) {
    path.push_back(cur);
    Pred next = TB[cur.time][cur.state];
    if (next.time == cur.time && next.state == cur.state) break;
    cur = next;
  }
  reverse(path.begin(), path.end());

  return {W[length - 1][C], path};
}
