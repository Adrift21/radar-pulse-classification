% TEST_BARKER_VISUALIZATION
% Sanity check for Barker phase-code generator. Produces:
%   - Figure 1: 6 Barker pulses (mix of B7/B11/B13).
%               Shows real part, instantaneous phase, STFT spectrogram.
%   - Figure 2: One example of each code (B7, B11, B13) side by side
%               with chip boundaries marked, so phase flips are obvious.
%   - Figure 3: LFM vs NLFM-quadratic vs Barker comparison
%               (instantaneous-frequency view, the "DNA" of each class).
%   - Figure 4: Barker at multiple SNRs (does signal survive at -10 dB?).
%
% Run from data_generation/matlab/ directory:
%   >> cd data_generation/matlab
%   >> addpath(genpath(pwd))
%   >> test_barker_visualization

clear; clc; close all;

addpath('config', 'signals', 'utils');

cfg = generation_config();

% ------------------------------------------------------------------
% Figure 1: 6 Barker pulses (clean)
% ------------------------------------------------------------------
figure('Name', 'Barker clean pulses', 'Position', [60 60 1300 800]);

code_count = struct('B7', 0, 'B11', 0, 'B13', 0);

for k = 1:6
    [s, p] = generate_barker(cfg);
    code_count.(p.code_name) = code_count.(p.code_name) + 1;

    t = (0:cfg.N-1) * cfg.Ts * 1e6;        % us

    subplot(6, 3, (k-1)*3 + 1);
    plot(t, real(s), 'b'); hold on; plot(t, imag(s), 'r');
    xlabel('t [\mus]'); ylabel('amp');
    title(sprintf('Barker #%d  [%s]  T=%.2f \\mus  T_c=%.0f ns  f_c=%.1f MHz', ...
          k, p.code_name, p.pulse_width_s*1e6, ...
          p.chip_duration_s*1e9, p.f_carrier_hz/1e6));
    legend('I','Q', 'Location', 'eastoutside'); grid on;

    subplot(6, 3, (k-1)*3 + 2);
    inst_phase = unwrap(angle(s));
    active = false(cfg.N,1);
    active(p.start_idx:p.stop_idx) = true;
    inst_phase_plot = inst_phase;
    inst_phase_plot(~active) = NaN;
    plot(t, inst_phase_plot, 'w', 'LineWidth', 1.2);
    xlabel('t [\mus]'); ylabel('phase [rad]');
    title('Unwrapped phase'); grid on;

    subplot(6, 3, (k-1)*3 + 3);
    spectrogram(s, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title('STFT spectrogram');
end

fprintf('Code mix in Figure 1: B7=%d, B11=%d, B13=%d (expect ~33%% each over many runs)\n', ...
        code_count.B7, code_count.B11, code_count.B13);

% ------------------------------------------------------------------
% Figure 2: One example of each code (B7, B11, B13) with chip
% boundaries marked
% ------------------------------------------------------------------
figure('Name', 'Barker codes side-by-side', 'Position', [120 60 1400 700]);

samples_by_code = struct();
for trial = 1:60
    [s_, p_] = generate_barker(cfg);
    if ~isfield(samples_by_code, p_.code_name)
        samples_by_code.(p_.code_name) = struct('s', s_, 'p', p_);
    end
    if numel(fieldnames(samples_by_code)) == 3, break; end
end

target_codes = {'B7', 'B11', 'B13'};
for idx = 1:3
    cn = target_codes{idx};
    s = samples_by_code.(cn).s;
    p = samples_by_code.(cn).p;
    t = (0:cfg.N-1) * cfg.Ts * 1e6;

    % Compute chip boundary times in microseconds
    chip_edges_idx = p.start_idx + (0:p.code_length) * p.chip_samples;
    chip_edges_us  = (chip_edges_idx - 1) * cfg.Ts * 1e6;

    % Real part with chip boundaries
    subplot(3, 2, (idx-1)*2 + 1);
    plot(t, real(s), 'b'); hold on;
    for ce = chip_edges_us
        xline(ce, 'g--', 'LineWidth', 0.5);
    end
    % Annotate chip values above signal
    for ci = 1:p.code_length
        x_mid = (chip_edges_us(ci) + chip_edges_us(ci+1)) / 2;
        y_pos = 1.1;
        chip_val = p.code_sequence(ci);
        if chip_val > 0
            text(x_mid, y_pos, '+', 'Color', [0.2 0.7 0.2], ...
                 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
        else
            text(x_mid, y_pos, '-', 'Color', [0.9 0.2 0.2], ...
                 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
        end
    end
    xlim([chip_edges_us(1) - 0.5, chip_edges_us(end) + 0.5]);
    ylim([-1.4, 1.4]);
    xlabel('t [\mus]'); ylabel('Re(s)');
    title(sprintf('%s real part  (T_c = %.0f ns,  f_c = %.1f MHz)', ...
          cn, p.chip_duration_s*1e9, p.f_carrier_hz/1e6));
    grid on;

    % Spectrogram zoomed to active region
    subplot(3, 2, (idx-1)*2 + 2);
    spectrogram(s, hamming(64), 56, 256, cfg.fs, 'yaxis', 'centered');
    title([cn, '  -  STFT spectrogram']);
end

% ------------------------------------------------------------------
% Figure 3: Class comparison - LFM vs NLFM-quadratic vs Barker
% ------------------------------------------------------------------
figure('Name', 'LFM vs NLFM vs Barker', 'Position', [180 60 1300 600]);

[s_lfm,    p_lfm]    = generate_lfm(cfg);

% Force a quadratic NLFM
s_nlfm = []; p_nlfm = [];
for trial = 1:50
    [s_, p_] = generate_nlfm(cfg);
    if strcmp(p_.variant, 'quadratic'), s_nlfm=s_; p_nlfm=p_; break; end
end

[s_brk,   p_brk]   = generate_barker(cfg);

triples = {
    s_lfm,  p_lfm,  'LFM (linear chirp)',                  'frequency';
    s_nlfm, p_nlfm, 'NLFM (quadratic)',                    'frequency';
    s_brk,  p_brk,  sprintf('Barker (%s)', p_brk.code_name), 'phase'
};

for k = 1:3
    s = triples{k,1}; p = triples{k,2}; name = triples{k,3};
    metric = triples{k,4};
    t = (0:cfg.N-1) * cfg.Ts * 1e6;

    subplot(3, 2, (k-1)*2 + 1);
    if strcmp(metric, 'frequency')
        inst_phase = unwrap(angle(s));
        inst_freq  = [0; diff(inst_phase)] / (2*pi*cfg.Ts);
        active = false(cfg.N,1); active(p.start_idx:p.stop_idx) = true;
        inst_freq(~active) = NaN;
        plot(t, inst_freq/1e6, 'w', 'LineWidth', 1.5);
        ylabel('f_{inst} [MHz]');
        title([name, '  -  Instantaneous frequency']);
    else
        % For Barker, phase is more revealing than frequency
        % (frequency would just show the carrier, missing the phase code)
        inst_phase = unwrap(angle(s));
        active = false(cfg.N,1); active(p.start_idx:p.stop_idx) = true;
        inst_phase(~active) = NaN;
        plot(t, inst_phase, 'w', 'LineWidth', 1.5);
        ylabel('phase [rad]');
        title([name, '  -  Unwrapped phase  (note kinks at chip boundaries)']);
    end
    xlabel('t [\mus]'); grid on;

    subplot(3, 2, (k-1)*2 + 2);
    spectrogram(s, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title([name, '  -  STFT spectrogram']);
end

% ------------------------------------------------------------------
% Figure 4: Barker at multiple SNRs
% ------------------------------------------------------------------
figure('Name', 'Barker at different SNRs', 'Position', [240 60 1200 700]);
snr_test = [-10, 0, 10, 20];

for i = 1:numel(snr_test)
    s_noisy = add_awgn(s_brk, snr_test(i), [p_brk.start_idx, p_brk.stop_idx]);

    subplot(numel(snr_test), 2, 2*i - 1);
    t = (0:cfg.N-1) * cfg.Ts * 1e6;
    plot(t, real(s_noisy));
    title(sprintf('Barker (%s), real part, SNR = %d dB', p_brk.code_name, snr_test(i)));
    xlabel('t [\mus]'); ylabel('amp'); grid on;

    subplot(numel(snr_test), 2, 2*i);
    spectrogram(s_noisy, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title(sprintf('Barker (%s) spectrogram, SNR = %d dB', p_brk.code_name, snr_test(i)));
end

fprintf('Tests complete. Inspect four figures.\n');
fprintf('  - Figure 1: 6 Barker pulses, code mix should approach 33%%/33%%/33%% over runs.\n');
fprintf('  - Figure 2: chip boundaries (green dashes) align with sign flips in real part.\n');
fprintf('  - Figure 3: LFM=line, NLFM=curve, Barker=phase steps (different metric used).\n');
fprintf('  - Figure 4: spectrogram should be a horizontal line that fades into noise.\n');
