% TEST_FRANK_VISUALIZATION
% Sanity check for Frank polyphase generator. Produces:
%   - Figure 1: 6 Frank pulses (mix of N=4/6/8).
%               Shows real part, wrapped phase, STFT spectrogram.
%   - Figure 2: Phase matrices for N=4 and N=8 as heatmaps,
%               with row-major flattened phase sequence below.
%   - Figure 3: 4-class comparison: LFM vs NLFM vs Barker vs Frank.
%   - Figure 4: Frank at multiple SNRs.
%
% Run from data_generation/matlab/ directory:
%   >> cd data_generation/matlab
%   >> addpath(genpath(pwd))
%   >> test_frank_visualization

clear; clc; close all;

addpath('config', 'signals', 'utils');

cfg = generation_config();

% ------------------------------------------------------------------
% Figure 1: 6 Frank pulses (clean)
% ------------------------------------------------------------------
figure('Name', 'Frank clean pulses', 'Position', [60 60 1300 800]);

N_count = struct('N4', 0, 'N6', 0, 'N8', 0);

for k = 1:6
    [s, p] = generate_frank(cfg);
    N_count.(sprintf('N%d', p.N)) = N_count.(sprintf('N%d', p.N)) + 1;

    t = (0:cfg.N-1) * cfg.Ts * 1e6;        % us

    subplot(6, 3, (k-1)*3 + 1);
    plot(t, real(s), 'b'); hold on; plot(t, imag(s), 'r');
    xlabel('t [\mus]'); ylabel('amp');
    title(sprintf('Frank #%d  N=%d  (%d chips)  T=%.2f \\mus  T_c=%.0f ns', ...
          k, p.N, p.num_chips, p.pulse_width_s*1e6, p.chip_duration_s*1e9));
    legend('I','Q', 'Location', 'eastoutside'); grid on;

    subplot(6, 3, (k-1)*3 + 2);
    % Wrapped phase (mod 2*pi) — better than unwrapped for Frank
    % because it shows the discrete phase levels directly.
    % Use white markers so they remain visible on MATLAB's dark theme.
    wrapped_phase = angle(s);
    active = false(cfg.N,1);
    active(p.start_idx:p.stop_idx) = true;
    wrapped_phase(~active) = NaN;
    plot(t, wrapped_phase, 'w.', 'MarkerSize', 4);
    yline(0, 'Color', [0.6 0.6 0.6], 'LineStyle', '--');
    yline(pi, 'Color', [0.6 0.6 0.6], 'LineStyle', '--');
    yline(-pi, 'Color', [0.6 0.6 0.6], 'LineStyle', '--');
    xlabel('t [\mus]'); ylabel('phase [rad]');
    title(sprintf('Wrapped phase (N=%d levels)', p.N));
    ylim([-pi-0.5, pi+0.5]); grid on;

    subplot(6, 3, (k-1)*3 + 3);
    spectrogram(s, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title('STFT spectrogram');
end

fprintf('N mix in Figure 1: N=4 -> %d, N=6 -> %d, N=8 -> %d (expect ~33%% each)\n', ...
        N_count.N4, N_count.N6, N_count.N8);

% ------------------------------------------------------------------
% Figure 2: Phase matrices and flattened sequences for N=4, N=8
% ------------------------------------------------------------------
figure('Name', 'Frank phase matrices', 'Position', [120 60 1200 700]);

% Force one example of N=4 and one of N=8
samples = struct();
for trial = 1:80
    [s_, p_] = generate_frank(cfg);
    key = sprintf('N%d', p_.N);
    if (p_.N == 4 || p_.N == 8) && ~isfield(samples, key)
        samples.(key) = struct('s', s_, 'p', p_);
    end
    if isfield(samples, 'N4') && isfield(samples, 'N8'), break; end
end

target_N = [4, 8];
for idx = 1:2
    Nval = target_N(idx);
    p = samples.(sprintf('N%d', Nval)).p;
    s = samples.(sprintf('N%d', Nval)).s;

    % Phase matrix as heatmap
    subplot(2, 2, (idx-1)*2 + 1);
    imagesc(p.phase_matrix);
    axis image; colormap(gca, hsv);
    colorbar('Ticks', [0, pi/2, pi, 3*pi/2], ...
             'TickLabels', {'0','\pi/2','\pi','3\pi/2'});
    xlabel('column n'); ylabel('row m');
    title(sprintf('N=%d phase matrix \\phi_{m,n} = (2\\pi/N)(m-1)(n-1)', Nval));
    set(gca, 'XTick', 1:Nval, 'YTick', 1:Nval);

    % Flattened phase sequence (row-major)
    subplot(2, 2, (idx-1)*2 + 2);
    chip_idx = 1:p.num_chips;
    stem(chip_idx, p.phase_sequence, 'filled', 'MarkerSize', 4);
    xlabel('chip index k'); ylabel('phase [rad]');
    title(sprintf('Row-major flattened sequence (%d chips)', p.num_chips));
    yticks([0, pi/2, pi, 3*pi/2, 2*pi]);
    yticklabels({'0','\pi/2','\pi','3\pi/2','2\pi'});
    ylim([-0.3, 2*pi + 0.3]); grid on;
end

% ------------------------------------------------------------------
% Figure 3: 4-class comparison: LFM vs NLFM vs Barker vs Frank
% ------------------------------------------------------------------
figure('Name', '4-class comparison', 'Position', [180 60 1300 800]);

[s_lfm, p_lfm] = generate_lfm(cfg);

% Force quadratic NLFM
s_nlfm = []; p_nlfm = [];
for trial = 1:50
    [s_, p_] = generate_nlfm(cfg);
    if strcmp(p_.variant, 'quadratic'), s_nlfm=s_; p_nlfm=p_; break; end
end

[s_brk, p_brk] = generate_barker(cfg);
[s_frk, p_frk] = generate_frank(cfg);

quads = {
    s_lfm,  p_lfm,  'LFM (linear)';
    s_nlfm, p_nlfm, 'NLFM (quadratic)';
    s_brk,  p_brk,  sprintf('Barker (%s)', p_brk.code_name);
    s_frk,  p_frk,  sprintf('Frank (N=%d)', p_frk.N)
};

for k = 1:4
    s = quads{k,1}; p = quads{k,2}; name = quads{k,3};

    subplot(4, 2, (k-1)*2 + 1);
    wrapped_phase = angle(s);
    active = false(cfg.N,1); active(p.start_idx:p.stop_idx) = true;
    wrapped_phase(~active) = NaN;
    plot((0:cfg.N-1)*cfg.Ts*1e6, wrapped_phase, 'w.', 'MarkerSize', 3);
    xlabel('t [\mus]'); ylabel('phase [rad]');
    title([name, '  -  Wrapped phase']);
    ylim([-pi-0.3, pi+0.3]); grid on;

    subplot(4, 2, (k-1)*2 + 2);
    spectrogram(s, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title([name, '  -  STFT spectrogram']);
end

% ------------------------------------------------------------------
% Figure 4: Frank at multiple SNRs
% ------------------------------------------------------------------
figure('Name', 'Frank at different SNRs', 'Position', [240 60 1200 700]);
snr_test = [-10, 0, 10, 20];

for i = 1:numel(snr_test)
    s_noisy = add_awgn(s_frk, snr_test(i), [p_frk.start_idx, p_frk.stop_idx]);

    subplot(numel(snr_test), 2, 2*i - 1);
    t = (0:cfg.N-1) * cfg.Ts * 1e6;
    plot(t, real(s_noisy));
    title(sprintf('Frank (N=%d), real part, SNR = %d dB', p_frk.N, snr_test(i)));
    xlabel('t [\mus]'); ylabel('amp'); grid on;

    subplot(numel(snr_test), 2, 2*i);
    spectrogram(s_noisy, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title(sprintf('Frank (N=%d) spectrogram, SNR = %d dB', p_frk.N, snr_test(i)));
end

fprintf('Tests complete. Inspect four figures.\n');
fprintf('  - Figure 1: 6 Frank pulses; wrapped phase shows N discrete levels.\n');
fprintf('  - Figure 2: phase matrices and row-major flattened sequences.\n');
fprintf('  - Figure 3: 4-class comparison; each class has distinct TF imza.\n');
fprintf('  - Figure 4: Frank should fade into noise at SNR = -10 dB.\n');