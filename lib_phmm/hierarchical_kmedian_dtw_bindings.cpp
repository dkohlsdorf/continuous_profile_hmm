#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <thread>
#include "hierarchical_kmedian_dtw.hpp"

namespace py = pybind11;

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// (frames, dims) numpy arrays -> Dataset, without going through one Python
// float object per value (a list-of-lists of floats is ~4x the raw size).
static Dataset to_dataset(const std::vector<Array> &sequences) {
  Dataset dataset;
  dataset.reserve(sequences.size());
  for (const Array &a : sequences) {
    if (a.ndim() != 2) throw py::value_error("each sequence must be a 2-d (frames, dims) array");
    auto r = a.unchecked<2>();
    Mat seq(r.shape(0), Vec(r.shape(1)));
    for (py::ssize_t i = 0; i < r.shape(0); i++)
      for (py::ssize_t d = 0; d < r.shape(1); d++) seq[i][d] = r(i, d);
    dataset.push_back(std::move(seq));
  }
  return dataset;
}

// DistanceManagement (in the header) stores a raw Dataset* and never takes
// ownership -- fine in C++ where the caller controls the dataset's
// lifetime, but not safe to expose directly to Python, where the dataset
// list could be garbage collected out from under it. This wrapper owns a
// copy of the dataset alongside the DistanceManagement that points into
// it, so the pointer stays valid for the wrapper's whole lifetime.
static int parse_norm(const std::string &norm) {
  if (norm == "path") return DTW_NORM_PATH;
  if (norm == "none") return DTW_NORM_NONE;
  if (norm == "length") return DTW_NORM_LENGTH;
  throw py::value_error("norm must be 'path', 'none' or 'length', got '" + norm + "'");
}

class DistanceManager {
public:
  DistanceManager(const std::vector<Array> &sequences, int warping_band, const std::string &norm)
      : dataset(to_dataset(sequences)), impl(&this->dataset, warping_band, parse_norm(norm)) {}

  double distance(int i, int j) {
    return impl.distance(i, j);
  }

  int size() {
    return impl.size();
  }

  // Divisive hierarchical k-medoids/k-median clustering under DTW distance
  // (see kmedoids() in hierarchical_kmedian_dtw.hpp). Recursively bisects
  // `instances`, stopping a branch once splitting it further would not
  // produce a denser (tighter) pair of children than the branch itself,
  // then returns the leaf medoids -- an empty `instances` clusters every
  // sequence in the dataset. restarts/tolerance default to the same
  // values hierarchical_kmedian_dtw.hpp itself defaults to.
  std::vector<int> kmedoids(std::vector<int> instances, int epochs, int restarts, double tolerance) {
    if (instances.empty()) {
      instances.resize(impl.size());
      for (int i = 0; i < impl.size(); i++) instances[i] = i;
    }
    std::set<int> medoids;
    ::kmedoids(impl, instances, medoids, epochs, 0.0, restarts, tolerance);
    return std::vector<int>(medoids.begin(), medoids.end());
  }

  // Same clustering as kmedoids(), but returns {medoid: leaf members}
  // (medoid included). Instances a splitting branch used as its anchor/
  // sample are not passed down to either child, so they appear in no leaf.
  std::map<int, std::vector<int>> kmedoids_leaves(std::vector<int> instances, int epochs, int restarts,
                                                  double tolerance) {
    if (instances.empty()) {
      instances.resize(impl.size());
      for (int i = 0; i < impl.size(); i++) instances[i] = i;
    }
    std::set<int> medoids;
    std::map<int, std::vector<int>> leaves;
    ::kmedoids(impl, instances, medoids, epochs, 0.0, restarts, tolerance, &leaves);
    return leaves;
  }

private:
  Dataset dataset;
  DistanceManagement impl;
};


// For each query, the index of its DTW-nearest medoid and that distance.
// Threaded over queries; the GIL is released while the DTWs run.
static std::pair<std::vector<int>, std::vector<double>> nearest(const std::vector<Array> &queries,
                                                                const std::vector<Array> &medoids,
                                                                int warping_band, const std::string &norm) {
  int norm_id = parse_norm(norm);
  Dataset q = to_dataset(queries), med = to_dataset(medoids);
  if (med.empty()) throw py::value_error("medoids must not be empty");
  std::vector<int> best(q.size(), -1);
  std::vector<double> best_dist(q.size(), INFINITY);
  {
    py::gil_scoped_release release;
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) {
      threads.emplace_back([&, t]() {
        for (size_t i = t; i < q.size(); i += n_threads) {
          for (size_t k = 0; k < med.size(); k++) {
            double d = dtw(q[i], med[k], warping_band, norm_id);
            if (d < best_dist[i]) { best_dist[i] = d; best[i] = (int)k; }
          }
        }
      });
    }
    for (auto &th : threads) th.join();
  }
  return {best, best_dist};
}


PYBIND11_MODULE(hierarchical_kmedian_dtw, m) {
  m.doc() = "Divisive hierarchical k-medoids clustering under DTW distance";

  m.def("dtw", [](Mat x, Mat y, int warping_band, const std::string &norm) {
          return dtw(x, y, warping_band, parse_norm(norm));
        },
        py::arg("x"), py::arg("y"), py::arg("warping_band"), py::arg("norm") = "path",
        "Sakoe-Chiba banded DTW distance between two sequences. norm: 'path' "
        "divides the summed cost by the warp-path length, 'length' by n + m, "
        "'none' returns the summed cost.");

  m.def("nearest", &nearest, py::arg("queries"), py::arg("medoids"), py::arg("warping_band"),
        py::arg("norm") = "path",
        "For each query sequence ((frames, dims) array), the index of its "
        "DTW-nearest medoid and that distance, as (indices, distances). "
        "Runs on all hardware threads.");

  py::class_<DistanceManager>(m, "DistanceManager")
    .def(py::init<const std::vector<Array> &, int, const std::string &>(), py::arg("dataset"), py::arg("warping_band"),
         py::arg("norm") = "path",
         "dataset: list of (frames, dims) numpy arrays. "
         "Pairwise DTW distances are computed lazily and cached. "
         "norm: 'path', 'length' or 'none' (see dtw()).")
    .def("distance", &DistanceManager::distance, py::arg("i"), py::arg("j"),
         "Cached DTW distance between dataset[i] and dataset[j].")
    .def("size", &DistanceManager::size)
    .def("kmedoids", &DistanceManager::kmedoids,
         py::arg("instances") = std::vector<int>{}, py::arg("epochs") = 20,
         py::arg("restarts") = KMEDOIDS_RESTARTS_DEFAULT, py::arg("tolerance") = DENSITY_TOLERANCE_DEFAULT,
         "Divisively cluster `instances` (dataset indices; empty = all) by "
         "DTW distance, splitting in two by k-medoids each level, recursing "
         "into a child only while it is denser (tighter around its medoid) "
         "than its parent -- no external distance/count threshold needed, "
         "the split count is entirely data-driven. `restarts` random "
         "(anchor, sample) attempts are tried per split and the densest is "
         "kept; `tolerance` is the fractional slack allowed below the "
         "parent's density before a branch stops (0.05 = a child up to 5% "
         "less dense than its parent still keeps splitting). Returns the "
         "resulting leaf medoids as dataset indices.")
    .def("kmedoids_leaves", &DistanceManager::kmedoids_leaves,
         py::arg("instances") = std::vector<int>{}, py::arg("epochs") = 20,
         py::arg("restarts") = KMEDOIDS_RESTARTS_DEFAULT, py::arg("tolerance") = DENSITY_TOLERANCE_DEFAULT,
         "Same clustering as kmedoids(), but returns {medoid: leaf members} "
         "(dataset indices, medoid included). Instances a splitting branch "
         "used as its anchor/sample are passed to neither child and appear "
         "in no leaf.");
}
