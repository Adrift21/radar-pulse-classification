% MAIN_GENERATE_DATASET
% Generate the synthetic radar pulse dataset for Module A.
%
% Produces 8 classes x cfg.samples_per_class samples (default 5000),
% giving 40,000 total clean baseband signals. Each sample gets:
%   - a class label (1..8)
%   - an intended SNR drawn from the SNR grid (-10..+20 dB, 2 dB step)
%   - a per-sample seed for reproducibility (master_seed + idx)
%
% Clean signals (no AWGN added) are written to HDF5 along with labels,
% intended SNR values, and metadata. AWGN is added in the Python
% pipeline at training time so the same clean signal can be augmented
% with different noise realizations.
%
% Run from data_generation/matlab/ directory:
%   >> cd data_generation/matlab
%   >> addpath(genpath(pwd))
%   >> main_generate_dataset

clear; clc;

% Path setup (idempotent)
addpath('config', 'signals', 'utils');

cfg = generation_config();

% ------------------------------------------------------------------
% Class -> generator function mapping
% Order MUST match cfg.class_names so labels are consistent.
% ------------------------------------------------------------------
generators = { ...
    @generate_lfm,        ...   % 1: LFM
    @generate_nlfm,       ...   % 2: NLFM
    @generate_barker,     ...   % 3: Barker
    @generate_frank,      ...   % 4: Frank
    @generate_polyphase,  ...   % 5: Polyphase (P1-P4)
    @generate_costas,     ...   % 6: Costas
    @generate_cw,         ...   % 7: CW
    @generate_stepped_fh  ...   % 8: SteppedFH
};

assert(numel(generators) == cfg.num_classes, ...
    'Generator count must match cfg.num_classes');

% Cross-check class names to catch ordering mistakes early
expected_names = {'LFM','NLFM','Barker','Frank','Polyphase', ...
                  'Costas','CW','SteppedFH'};
for c = 1 : cfg.num_classes
    if ~strcmp(cfg.class_names{c}, expected_names{c})
        warning('main_generate_dataset:classOrderMismatch', ...
            'cfg.class_names{%d}=''%s'' but expected ''%s''', ...
            c, cfg.class_names{c}, expected_names{c});
    end
end

N_total = cfg.total_samples;
L       = cfg.N;

% ------------------------------------------------------------------
% Pre-allocate output arrays
% ------------------------------------------------------------------
signals         = complex(zeros(L, N_total, 'double'));   % (L, N)
labels          = zeros(N_total, 1, 'uint8');             % 1-based here
snr_db          = zeros(N_total, 1, 'double');
pulse_widths_us = zeros(N_total, 1, 'double');

% ------------------------------------------------------------------
% Generation loop (sequential per class)
% ------------------------------------------------------------------
fprintf('=================================================================\n');
fprintf('  Generating dataset: %d classes x %d samples = %d total\n', ...
        cfg.num_classes, cfg.samples_per_class, N_total);
fprintf('  Sample rate     : %.1f MHz\n', cfg.fs/1e6);
fprintf('  Signal length   : %d samples (%.2f us)\n', L, L*cfg.Ts*1e6);
fprintf('  SNR grid        : %d points from %d to %d dB (step %d)\n', ...
        numel(cfg.snr_db_grid), cfg.snr_db_min, cfg.snr_db_max, ...
        cfg.snr_db_step);
fprintf('  Master seed     : %d (per-sample seed = master + idx)\n', ...
        cfg.master_seed);
fprintf('=================================================================\n');

t_start_total = tic;

for class_idx = 1 : cfg.num_classes
    cls_name = cfg.class_names{class_idx};
    gen_fn   = generators{class_idx};

    fprintf('\n[%d/%d] Generating class ''%s''...\n', ...
            class_idx, cfg.num_classes, cls_name);

    t_class = tic;

    for k = 1 : cfg.samples_per_class

        % Global sample index across all classes (1-based)
        global_idx = (class_idx - 1) * cfg.samples_per_class + k;

        % Per-sample seed for reproducibility
        rng(cfg.master_seed + global_idx, 'twister');

        % Generate clean signal
        [s, ~] = gen_fn(cfg);

        % Sample SNR from the discrete grid
        snr_choice = cfg.snr_db_grid(randi(numel(cfg.snr_db_grid)));

        % Determine actual pulse width: regenerate params via signal
        % isn't trivial; instead compute from active region (numel where
        % |s|>0). Since active region is contiguous after pad_signal,
        % count non-zero magnitudes.
        active_count = sum(abs(s) > 0);
        pw_us = active_count * cfg.Ts * 1e6;

        % Store
        signals(:, global_idx)        = s;
        labels(global_idx)            = uint8(class_idx);   % 1-based
        snr_db(global_idx)            = snr_choice;
        pulse_widths_us(global_idx)   = pw_us;

        % Progress reporting
        if cfg.verbose && mod(k, cfg.progress_every) == 0
            elapsed_class = toc(t_class);
            rate = k / elapsed_class;
            eta_class = (cfg.samples_per_class - k) / rate;
            fprintf('  %s: %d/%d   (%.0f samp/s, ETA %.0fs for this class)\n', ...
                    cls_name, k, cfg.samples_per_class, rate, eta_class);
        end
    end

    fprintf('  %s: done in %.1fs\n', cls_name, toc(t_class));
end

elapsed_total = toc(t_start_total);
fprintf('\n=================================================================\n');
fprintf('  All %d samples generated in %.1f s\n', N_total, elapsed_total);
fprintf('=================================================================\n');

% ------------------------------------------------------------------
% Sanity checks before writing
% ------------------------------------------------------------------
assert(all(labels >= 1 & labels <= cfg.num_classes), ...
    'Some labels are out of expected range');

% Per-class count must equal samples_per_class
class_counts = histcounts(labels, 0.5 : cfg.num_classes + 0.5);
assert(all(class_counts == cfg.samples_per_class), ...
    'Class counts do not match samples_per_class');

% SNR values must lie on the grid
assert(all(ismember(snr_db, cfg.snr_db_grid)), ...
    'Some SNR values are off the configured grid');

fprintf('\nSanity checks passed:\n');
fprintf('  - All labels in [1, %d]\n', cfg.num_classes);
fprintf('  - Each class has %d samples\n', cfg.samples_per_class);
fprintf('  - All SNR values are on the configured grid\n');

% ------------------------------------------------------------------
% Write to HDF5
% ------------------------------------------------------------------
fprintf('\nWriting HDF5...\n');
save_dataset_h5(cfg.output_file, signals, labels, snr_db, ...
                pulse_widths_us, cfg);

fprintf('\nDone. Dataset written to:\n  %s\n\n', cfg.output_file);
