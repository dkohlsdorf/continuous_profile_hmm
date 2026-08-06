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
    .def("__repr__", [](const Gaussian& g) {
      ostringstream os;
      os << g;
      return os.str();
    });

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
        py::arg("sequence"), py::arg("phmm"),
        "Run Viterbi decoding. Returns (log2_score, path) where path is a list of Pred.");
}
