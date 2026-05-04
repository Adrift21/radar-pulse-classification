function cfg = generation_config()
% GENERATION_CONFIG  Central configuration for synthetic radar dataset.
%
% Returns a struct with all parameters used by the data generation
% pipeline. All decisions tracked in docs/decisions.md (2026-05-04 entries).
%
% Usage:
%   cfg = generation_config();
%   fs = cfg.fs;
%
% Author: Kaan Emre Evci
% Project: Radar Pulse Classification

    % ------------------------------------------------------------------
    % Reproducibility
    % ------------------------------------------------------------------
    cfg.master_seed = 42;
    rng(cfg.master_seed, 'twister');  % Set MATLAB global RNG

    % ------------------------------------------------------------------
    % Sampling parameters
    % ------------------------------------------------------------------
    cfg.fs = 100e6;              % Sample rate [Hz]  -> 100 MHz
    cfg.Ts = 1 / cfg.fs;         % Sample period [s] -> 10 ns
    cfg.N  = 2048;               % Fixed signal length [samples]
                                 % => 20.48 us window @ 100 MHz

    % ------------------------------------------------------------------
    % Pulse parameters
    % ------------------------------------------------------------------
    cfg.pulse_width_us = [1, 20];     % [min, max] in microseconds
    cfg.pulse_width_s  = cfg.pulse_width_us * 1e-6;

    % Padding strategy: 'random' | 'left' | 'right' | 'center'
    cfg.padding_strategy = 'random';

    % ------------------------------------------------------------------
    % Carrier / IF parameters
    % ------------------------------------------------------------------
    % Working in complex baseband (I/Q). Carrier frequency is the offset
    % from baseband; classes that need a carrier (CW, stepped) use this.
    % Range chosen to stay well below Nyquist (fs/2 = 50 MHz).
    cfg.fc_range_hz = [1e6, 20e6];     % [min, max] carrier frequency [Hz]

    % LFM/NLFM specific
    cfg.lfm_bandwidth_hz = [5e6, 20e6]; % chirp bandwidth range [Hz]

    % ------------------------------------------------------------------
    % Class definitions
    % ------------------------------------------------------------------
    cfg.class_names = { ...
        'LFM',         ...   % 1: Linear Frequency Modulation
        'NLFM',        ...   % 2: Nonlinear FM (e.g., quadratic)
        'Barker',      ...   % 3: Barker phase code (B7/B11/B13 mix)
        'Frank',       ...   % 4: Frank polyphase code
        'Polyphase',   ...   % 5: P1/P2/P3/P4 unified class (TBD)
        'Costas',      ...   % 6: Costas frequency hopping
        'CW',          ...   % 7: Continuous Wave (constant freq)
        'SteppedFH'    ...   % 8: Stepped frequency / Frequency hopping
    };
    cfg.num_classes = numel(cfg.class_names);

    cfg.samples_per_class = 5000;
    cfg.total_samples = cfg.num_classes * cfg.samples_per_class;  % 40,000

    % ------------------------------------------------------------------
    % SNR configuration
    % ------------------------------------------------------------------
    cfg.snr_db_min  = -10;
    cfg.snr_db_max  =  20;
    cfg.snr_db_step =   2;
    cfg.snr_db_grid = cfg.snr_db_min : cfg.snr_db_step : cfg.snr_db_max;
    % => [-10, -8, -6, ..., 18, 20], 16 points

    % AWGN added AFTER padding to the full N-sample signal.
    % SNR is computed using signal power on the active (non-zero) region.

    % ------------------------------------------------------------------
    % Signal normalization
    % ------------------------------------------------------------------
    % All clean signals are normalized to unit average power on the
    % active region BEFORE AWGN is added. This makes SNR specification
    % unambiguous: noise variance = 1 / 10^(SNR/10).
    cfg.normalize_signal_power = true;

    % ------------------------------------------------------------------
    % Output paths
    % ------------------------------------------------------------------
    % Code lives in data_generation/matlab/, output goes to
    % data_generation/synthetic_samples/. We resolve relative to this
    % file's location so the script works regardless of MATLAB's pwd.
    this_file_dir = fileparts(mfilename('fullpath'));   % .../matlab/config
    matlab_root   = fileparts(this_file_dir);           % .../matlab
    dg_root       = fileparts(matlab_root);             % .../data_generation
    cfg.output_dir  = fullfile(dg_root, 'synthetic_samples');
    cfg.output_file = fullfile(cfg.output_dir, 'dataset.h5');
    cfg.dataset_version = '0.1.0';

    % ------------------------------------------------------------------
    % Logging
    % ------------------------------------------------------------------
    cfg.verbose = true;
    cfg.progress_every = 500;   % Print progress every N samples per class
end
