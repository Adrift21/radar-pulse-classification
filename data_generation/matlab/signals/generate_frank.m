    function [signal, params] = generate_frank(cfg)
% GENERATE_FRANK  Synthesize a single Frank polyphase-coded pulse.
%
% Frank codes are an N-phase generalization of binary phase codes
% (Barker). The phase of the (m,n)-th chip in an N x N matrix is
%
%   phi(m,n) = (2*pi / N) * (m-1) * (n-1),   m,n = 1..N
%
% The code sequence is obtained by reading the matrix row by row,
% giving N^2 chips total. For example, N=4 produces 16 chips with
% phases drawn from {0, pi/2, pi, 3*pi/2}.
%
% N is selected at random from {4, 6, 8} with equal probability,
% giving 16, 36, or 64 chips per pulse.
%
% Signal model (zero carrier, complex baseband)
% ---------------------------------------------
%   s(t) = exp(j * phi_chip(k)),    k = floor(t / Tc) + 1
%
% No additional carrier modulation: f_c = 0. The Frank code's TF
% signature is dominated by the phase matrix structure; adding a
% carrier would just shift the spectrum without adding distinguishing
% information. See decisions.md (2026-05-04 Frank entry).
%
% Inputs
% ------
%   cfg : struct from generation_config()
%
% Outputs
% -------
%   signal : (cfg.N x 1) complex double, unit-power on active region
%   params : struct
%       .class_name      = 'Frank'
%       .N               = matrix size (4, 6, or 8)
%       .num_chips       = N^2
%       .phase_matrix    = (N x N) double, phases in radians
%       .phase_sequence  = (1 x N^2) double, row-major flatten in radians
%       .pulse_width_s   = total pulse duration T in seconds
%       .num_active      = pulse length in samples
%       .chip_duration_s = Tc = T / N^2
%       .chip_samples    = round(Tc * fs)
%       .start_idx       = index where pulse begins (1-based)
%       .stop_idx        = index where pulse ends (1-based, inclusive)
%
% Notes
% -----
% - Uses cfg.polyphase_pulse_width_s (default [4, 20] us, shared with
%   P1-P4) instead of the generic cfg.pulse_width_s, to guarantee
%   >= 6 samples/chip even at N=8 (64 chips) and the minimum pulse
%   width.
% - The phase matrix is exact (multiples of 2*pi/N); chip flips are
%   instantaneous, like Barker. No pulse-shaping filter (consistent
%   with Barker's rectangular chip decision).

    % ------------------------------------------------------------------
    % 1) Select matrix size N at random (equal probability)
    % ------------------------------------------------------------------
    N_options = [4, 6, 8];
    N = N_options(randi(numel(N_options)));
    num_chips = N * N;

    % ------------------------------------------------------------------
    % 2) Build N x N Frank phase matrix
    %    phi(m,n) = (2*pi / N) * (m-1) * (n-1)
    % ------------------------------------------------------------------
    [m_idx, n_idx] = ndgrid(0 : N-1, 0 : N-1);   % 0-based for the formula
    phase_matrix = (2*pi / N) * m_idx .* n_idx;

    % Flatten row-major (MATLAB stores column-major, so transpose first)
    phase_sequence = reshape(phase_matrix.', 1, []);   % (1 x N^2)

    % ------------------------------------------------------------------
    % 3) Sample pulse width from the shared polyphase range
    %    (renamed from frank_pulse_width_s when P1-P4 added)
    % ------------------------------------------------------------------
    pw_min = cfg.polyphase_pulse_width_s(1);
    pw_max = cfg.polyphase_pulse_width_s(2);
    T = pw_min + (pw_max - pw_min) * rand();

    % ------------------------------------------------------------------
    % 4) Compute chip duration & sample count
    % ------------------------------------------------------------------
    Tc = T / num_chips;
    chip_samples = max(1, round(Tc * cfg.fs));
    num_active = chip_samples * num_chips;
    num_active = min(num_active, cfg.N);

    % If clipping happened (very rare), truncate code length
    full_chips_fit = floor(num_active / chip_samples);
    if full_chips_fit < num_chips
        phase_sequence = phase_sequence(1:full_chips_fit);
        num_chips = full_chips_fit;
        num_active = chip_samples * num_chips;
    end

    % ------------------------------------------------------------------
    % 5) Expand phase sequence to per-sample phase array
    % ------------------------------------------------------------------
    phase_per_sample = repelem(phase_sequence(:), chip_samples);
    % phase_per_sample is now (num_active x 1)

    % ------------------------------------------------------------------
    % 6) Build complex baseband signal (no carrier)
    %    s = exp(j * phi_chip(k))
    % ------------------------------------------------------------------
    pulse = exp(1j * phase_per_sample);

    % ------------------------------------------------------------------
    % 7) Normalize active region to unit average power
    %    (phase-only signal already has |s|^2 = 1, so this is a no-op
    %    in the ideal case; we keep the call for consistency.)
    % ------------------------------------------------------------------
    if cfg.normalize_signal_power
        p = mean(abs(pulse).^2);
        if p > 0
            pulse = pulse / sqrt(p);
        end
    end

    % ------------------------------------------------------------------
    % 8) Place pulse inside fixed-length frame
    % ------------------------------------------------------------------
    [signal, start_idx, stop_idx] = pad_signal(pulse, cfg.N, ...
                                               cfg.padding_strategy);

    % ------------------------------------------------------------------
    % 9) Pack params
    % ------------------------------------------------------------------
    params.class_name      = 'Frank';
    params.N               = N;
    params.num_chips       = num_chips;
    params.phase_matrix    = phase_matrix;
    params.phase_sequence  = phase_sequence;
    params.pulse_width_s   = T;
    params.num_active      = num_active;
    params.chip_duration_s = Tc;
    params.chip_samples    = chip_samples;
    params.start_idx       = start_idx;
    params.stop_idx        = stop_idx;
end
