#include<vector>
#include<cmath>
#include<cstdlib>
#include<algorithm>
#include<random>
#include<utility>
#include<set>
#include<iostream>

using namespace std;

using Vec = vector<double>; // duplicate
using Mat = vector<Vec>;    // duplicate
using Dataset = vector<Mat>;

inline double euc(Vec &x, Vec &y) {
  int n = x.size();
  double distance = 0.0;
  for(int i = 0; i < n; i++) {
    distance += pow(x[i] - y[i], 2);
  }
  return distance;
}

inline double min3(double x, double y, double z) {
  double min = x;
  if(y < min) min = y;
  if(z < min) min = z;
  return min;
}

inline double dtw(Mat &x, Mat &y, int warping_band) {
  int n = x.size();
  int m = y.size();

  Mat dp = Mat(n + 1, Vec(m + 1, INFINITY));
  dp[0][0] = 0.0;

  int w = max(warping_band, abs(n - m));

  for(int i = 1; i <= n; i++) {
    int j_start = max(1, i - w);
    int j_end = min(m, i + w);
    for(int j = j_start; j <= j_end; j++) {
      dp[i][j] = min3(dp[i - 1][j],
		      dp[i - 1][j - 1],
		      dp[i][j - 1]);
      dp[i][j] += euc(x[i - 1], y[j - 1]);
    }
  }

  return dp[n][m];
}

inline int instance_pair_hash(int i, int j, int n_instances) {
  return i * n_instances + j;
}

class DistanceManagement {
public:
  DistanceManagement(Dataset *dataset, int warping_band):dataset(dataset), warping_band(warping_band) {
    store = Mat(dataset->size(), Vec(dataset->size(), INFINITY));
  }

  double distance(int i, int j) {
    if(isinf(store[i][j])) {
      store[i][j] = dtw((*dataset)[i], (*dataset)[j], warping_band);
      store[j][i] = store[i][j];
    }
    return store[i][j];
  }

  int size() {
    return dataset -> size();
  }

private:
  int warping_band;
  Dataset *dataset;
  Mat store;
};


inline int sample_medoid(DistanceManagement &x, vector<int> &instances, int anchor, mt19937 &gen) {
  int n = instances.size();
  Vec cum_distances = Vec(n, 0.0);
  for(int i = 0; i < n; i++) {
    cum_distances[i] = x.distance(anchor, instances[i]);
    if(i > 0) cum_distances[i] += cum_distances[i - 1];
  }
  std::uniform_real_distribution<> dis(0.0, cum_distances[n - 1]);
  double r = dis(gen);
  for(int i = 0; i < n; i++) {
    if(anchor != instances[i]) {
      if(r < cum_distances[i]) return instances[i];
    }
  }
  return instances[n - 1];
}

inline pair<int, double> select_medoid(int medoid, double current_avg, vector<int> &inst, DistanceManagement &x) {
  int new_medoid = medoid;
  double min_avg_dist = current_avg;
  int n = (int)inst.size();
  for(int i = 0; i < n; i++) {
    double total = 0.0;
    for(int j = 0; j < n; j++) {
      total += x.distance(inst[i], inst[j]);
    }
    double avg = total / n;
    if(avg < min_avg_dist) {
      min_avg_dist = avg;
      new_medoid = inst[i];
    }
  }
  return make_pair(new_medoid, min_avg_dist);
}

inline void kmedoids(DistanceManagement &x, vector<int> &instances, set<int> &medoids, int epochs, double threshold) {
  int n = instances.size();
  if(n <= 2) {
    medoids.insert(instances.begin(), instances.end());
    return;
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<std::mt19937::result_type> dist(0, n - 1);
  int anchor = instances[dist(gen)];
  int sample = sample_medoid(x, instances, anchor, gen);

  double anchor_dist = INFINITY;
  double sample_dist = INFINITY;
  vector<int> anchor_inst, sample_inst;
  for(int epoch = 0; epoch < epochs; epoch++) {
    anchor_inst.clear();
    sample_inst.clear();

    double total_anchor = 0.0;
    double total_sample = 0.0;
    for(int i = 0; i < n; i++) {
      if(instances[i] != anchor and instances[i] != sample) {
	double d_anchor = x.distance(anchor, instances[i]);
	double d_sample = x.distance(sample, instances[i]);
	total_anchor += d_anchor;
	total_sample += d_sample;
	if(d_anchor < d_sample) {
	  anchor_inst.push_back(instances[i]);
	} else {
	  sample_inst.push_back(instances[i]);
	}
      }
    }

    double avg_anchor = total_anchor / max(1, (int)anchor_inst.size());
    double avg_sample = total_sample / max(1, (int)sample_inst.size());
    auto new_anchor = select_medoid(anchor, avg_anchor, anchor_inst, x);
    auto new_sample = select_medoid(sample, avg_sample, sample_inst, x);

    anchor_dist = new_anchor.second;
    sample_dist = new_sample.second;

    if(anchor == new_anchor.first and sample == new_sample.first) break;

    anchor = new_anchor.first;
    sample = new_sample.first;
  }

  if(anchor_dist >= threshold) {
    kmedoids(x, anchor_inst, medoids, epochs, threshold);
  } else {
    medoids.insert(anchor);
  }
  if(sample_dist >= threshold) {
    kmedoids(x, sample_inst, medoids, epochs, threshold);
  } else {
    medoids.insert(sample);
  }
}

