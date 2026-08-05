import profile_hmm as p


if __name__ == '__main__':
    dummy     = [0.0, 0.0, 0.0]
    vdummy    = [1.0, 1.0, 1.0]
    var_tight = [0.01, 0.01, 0.01]
    
    # sub-HMM 0: ascending pattern  [1,1,1] → [2,2,2]
    mean_a1 = [1.0, 1.0, 1.0]
    mean_a2 = [2.0, 2.0, 2.0]
    
    # sub-HMM 1: descending pattern [9,9,9] → [8,8,8]
    mean_b1 = [9.0, 9.0, 9.0]
    mean_b2 = [8.0, 8.0, 8.0]
    
    pdf = [
        [p.Gaussian(dummy, vdummy), p.Gaussian(mean_a1, var_tight), p.Gaussian(mean_a2, var_tight)],
        [p.Gaussian(dummy, vdummy), p.Gaussian(mean_b1, var_tight), p.Gaussian(mean_b2, var_tight)],
    ]
    
    trans = p.FlankTransitions(
        nn=0.001, nb=0.999,
        ec=0.5,   ej=0.5,
        jj=0.8,   jb=0.2,
        cc=0.9,
        b_to_hmm=[[0.25, 0.5, 0.25],
                   [0.25, 0.5, 0.25]],
        mm=[[0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1]],
    )
    
    hmm = p.ProfileHMM(pdf, trans)
    
    # Expected path: N(0), B(0), M[0][1](1), M[0][1](2) [self-transition], M[0][2](3), E(3), J(4), B(4), M[1][1](5), M[1][2](6), E(6), C(6), C(7), C(8)
    sequence = [
        [0.00, 0.00, 0.00],
        [1.05, 1.05, 1.05],
        [1.04, 1.06, 1.02],
        [2.03, 1.97, 2.01],
        [0.00, 0.00, 0.00],
        [8.98, 9.02, 9.01],
        [8.03, 7.97, 7.99],
        [0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00],
    ]
    
    score, path = p.viterbi(sequence, hmm)
    
    print(f"Viterbi score: {score}")
    print("Viterbi path:")
    n_models = len(hmm.pdf)
    n_states = len(hmm.pdf[0])
    for step in path:
        print(f"  t={step.time}  {p.state_name(step.state, n_models, n_states)}")
