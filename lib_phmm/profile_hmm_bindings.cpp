#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include "profile_hmm.hpp"

namespace py = pybind11;

PYBIND11_MODULE(profile_hmm, m) {
  m.doc() = "Accelerated Profile HMM search for continuous data";

  m.attr("MATCH_STATE") = MATCH_STATE;

  py::enum_<FlankState>(m, "FlankState")
    .value("N", FlankState::N)
    .value("B", FlankState::B)
    .value("E", FlankState::E)
    .value("C", FlankState::C)
    .value("J", FlankState::J)
    .export_values();

  py::class_<Gaussian>(m, "Gaussian")
    .def(py::init<Vec&, Vec&>(), py::arg("mean"), py::arg("variance"))
    .def("ll", &Gaussian::ll, py::arg("x"),
         "Log-likelihood of observation x under this Gaussian")
    .def(py::pickle(
      [](const Gaussian& g) {
        return py::make_tuple(g.get_mean(), g.get_variance());
      },
      [](py::tuple t) {
        if (t.size() != 2) throw std::runtime_error("Invalid Gaussian pickle state");
        Vec mean = t[0].cast<Vec>();
        Vec variance = t[1].cast<Vec>();
        return Gaussian(mean, variance);
      }
    ))
    .def("__repr__", [](const Gaussian& g) {
      ostringstream os;
      os << g;
      return os.str();
    });

  py::class_<MixtureModel>(m, "MixtureModel")
    .def(py::init<vector<Gaussian>, Vec>(), py::arg("components"), py::arg("log_weights"),
         "components: list of Gaussian. log_weights: mixture weights in log space "
         "(not raw probabilities -- e.g. np.log(sklearn GaussianMixture.weights_)), "
         "same length as components.")
    .def("ll", &MixtureModel::ll, py::arg("x"),
         "Log-likelihood of observation x under this mixture (log-sum-exp over components).")
    .def(py::pickle(
      [](const MixtureModel& mm) {
        return py::make_tuple(mm.get_components(), mm.get_log_weights());
      },
      [](py::tuple t) {
        if (t.size() != 2) throw std::runtime_error("Invalid MixtureModel pickle state");
        auto components = t[0].cast<vector<Gaussian>>();
        auto log_weights = t[1].cast<Vec>();
        return MixtureModel(components, log_weights);
      }
    ));

  py::class_<FlankTransitions>(m, "FlankTransitions")
    .def(py::init<float, float, float, float, float, float, float, vector<vector<float>>, vector<vector<float>>>(),
         py::arg("nn"), py::arg("nb"),
         py::arg("ec"), py::arg("ej"),
         py::arg("jj"), py::arg("jb"),
         py::arg("cc"), py::arg("b_to_hmm"), py::arg("mm"),
         "All probabilities are given as raw probabilities; log2 is applied internally.")
    .def_readwrite("nn", &FlankTransitions::nn)
    .def_readwrite("nb", &FlankTransitions::nb)
    .def_readwrite("ec", &FlankTransitions::ec)
    .def_readwrite("ej", &FlankTransitions::ej)
    .def_readwrite("jj", &FlankTransitions::jj)
    .def_readwrite("jb", &FlankTransitions::jb)
    .def_readwrite("cc", &FlankTransitions::cc)
    .def_readwrite("b_to_hmm", &FlankTransitions::b_to_hmm)
    .def_readwrite("mm", &FlankTransitions::mm)
    .def(py::pickle(
      [](const FlankTransitions& t) {
        // nn/nb/.../mm are stored as log2(probability) -- exp2 recovers
        // the raw probabilities the constructor expects, so setstate can
        // reuse it unchanged instead of duplicating its log2 logic.
        auto exp2_mat = [](const vector<vector<float>>& mat) {
          vector<vector<float>> out;
          out.reserve(mat.size());
          for (auto& row : mat) {
            vector<float> r;
            r.reserve(row.size());
            for (float v : row) r.push_back(exp2(v));
            out.push_back(std::move(r));
          }
          return out;
        };
        return py::make_tuple(
          exp2(t.nn), exp2(t.nb),
          exp2(t.ec), exp2(t.ej),
          exp2(t.jj), exp2(t.jb),
          exp2(t.cc),
          exp2_mat(t.b_to_hmm), exp2_mat(t.mm)
        );
      },
      [](py::tuple t) {
        if (t.size() != 9) throw std::runtime_error("Invalid FlankTransitions pickle state");
        return FlankTransitions(
          t[0].cast<float>(), t[1].cast<float>(),
          t[2].cast<float>(), t[3].cast<float>(),
          t[4].cast<float>(), t[5].cast<float>(),
          t[6].cast<float>(),
          t[7].cast<vector<vector<float>>>(),
          t[8].cast<vector<vector<float>>>()
        );
      }
    ))
    .def("__repr__", [](const FlankTransitions& t) {
      ostringstream os;
      os << t;
      return os.str();
    });

  py::class_<ProfileHMM>(m, "ProfileHMM")
    .def(py::init<vector<vector<Gaussian>>&, FlankTransitions&>(),
         py::arg("pdf"), py::arg("trans"))
    .def_readwrite("pdf", &ProfileHMM::pdf)
    .def_readwrite("trans", &ProfileHMM::trans)
    .def(py::pickle(
      [](const ProfileHMM& hmm) {
        return py::make_tuple(hmm.pdf, hmm.trans);
      },
      [](py::tuple t) {
        if (t.size() != 2) throw std::runtime_error("Invalid ProfileHMM pickle state");
        auto pdf = t[0].cast<vector<vector<Gaussian>>>();
        auto trans = t[1].cast<FlankTransitions>();
        return ProfileHMM(pdf, trans);
      }
    ))
    .def("__repr__", [](const ProfileHMM& hmm) {
      ostringstream os;
      os << hmm;
      return os.str();
    });

  py::class_<Pred>(m, "Pred")
    .def(py::init<>())
    .def_readwrite("time", &Pred::time)
    .def_readwrite("state", &Pred::state)
    .def("__repr__", [](const Pred& p) {
      return "Pred(time=" + to_string(p.time) + ", state=" + to_string(p.state) + ")";
    });

  m.def("match_state", &match_state,
        py::arg("i"), py::arg("hmm_idx"), py::arg("n_match_states"),
        "Return the flat state index for match state k of sub-HMM n.");

  m.def("state_name", &state_name,
        py::arg("state"), py::arg("n_models"), py::arg("match_state_per_model"),
        "Return a human-readable name for a state index.");

  m.def("viterbi", &viterbi,
        py::arg("sequence"), py::arg("phmm"), py::arg("noise_pdf") = py::none(),
        "Run Viterbi decoding. Returns (log2_score, path) where path is a list of Pred. "
        "noise_pdf: optional MixtureModel giving N/J/C a real emission model (competing "
        "against match states) instead of the default silent/transition-only behavior "
        "-- see profile_hmm.hpp's viterbi() comment.");
}
