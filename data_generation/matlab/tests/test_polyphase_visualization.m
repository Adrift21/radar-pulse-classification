% TEST_POLYPHASE_VISUALIZATION
% Sanity check for P1/P2/P3/P4 polyphase generator. Produces:
%   - Figure 1: 8 polyphase pulses (mix of P1/P2/P3/P4 and N=4/6/8).
%               Shows real part, wrapped phase, STFT spectrogram.
%   - Figure 2: Force one of each sub-code (P1, P2, P3, P4) side by side
%               for direct comparison: phase plot, stem of phase sequence,
%               and spectrogram.
%   - Figure 3: 5-class comparison: LFM vs NLFM vs Barker vs Frank vs Polyphase.
%               Especially highlights P3 looking similar to LFM.
%   - Figure 4: Polyphase at multiple SNRs.
%
% Run from data_generation/matlab/ directory:
%   >> cd data_generation/matlab
%   >> addpath(genpath(pwd))
%   >> test_polyphase_visualization

clear; clc; close all;

addpath('config', 'signals', 'utils');

cfg = generation_config();

% ------------------------------------------------------------------
% Figure 1: 8 polyphase pulses (clean) — broad sample of sub-codes & N
% ------------------------------------------------------------------
figure('Name', 'Polyphase clean pulses', 'Position', [60 60 1300 900]);

subcode_count = struct('P1', 0, 'P2', 0, 'P3', 0, 'P4', 0);

for k = 1:8
    [s, p] = generate_polyphase(cfg);
    subcode_count.(p.subcode) = subcode_count.(p.subcode) + 1;

    t = (0:cfg.N-1) * cfg.Ts * 1e6;

    subplot(8, 3, (k-1)*3 + 1);
    plot(t, real(s), 'b'); hold on; plot(t, imag(s), 'r');
    xlabel('t [\mus]'); ylabel('amp');
    title(sprintf('#%d  [%s, N=%d, %d chips]  T=%.2f \\mus  T_c=%.0f ns', ...
          k, p.subcode, p.N, p.num_chips, ...
          p.pulse_width_s*1e6, p.chip_duration_s*1e9));
    legend('I','Q', 'Location', 'eastoutside'); grid on;

    subplot(8, 3, (k-1)*3 + 2);
    wrapped_phase = angle(s);
    active = false(cfg.N,1);
    active(p.start_idx:p.stop_idx) = true;
    wrapped_phase(~active) = NaN;
    plot(t, wrapped_phase, 'w.', 'MarkerSize', 4);
    xlabel('t [\mus]'); ylabel('phase [rad]');
    title(sprintf('Wrapped phase (%s)', p.subcode));
    ylim([-pi-0.3, pi+0.3]); grid on;

    subplot(8, 3, (k-1)*3 + 3);
    spectrogram(s, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title('STFT spectrogram');
end

fprintf('Subcode mix in Figure 1: P1=%d, P2=%d, P3=%d, P4=%d\n', ...
        subcode_count.P1, subcode_count.P2, ...
        subcode_count.P3, subcode_count.P4);

% ------------------------------------------------------------------
% Figure 2: Force one example of each sub-code for direct comparison
% ------------------------------------------------------------------
figure('Name', 'P1 vs P2 vs P3 vs P4', 'Position', [120 30 1300 900]);

samples = struct();
target_subcodes = {'P1', 'P2', 'P3', 'P4'};

for trial = 1:200
    [s_, p_] = generate_polyphase(cfg);
    if ~isfield(samples, p_.subcode)
        samples.(p_.subcode) = struct('s', s_, 'p', p_);
    end
    if numel(fieldnames(samples)) == 4, break; end
end

for idx = 1:4
    cn = target_subcodes{idx};
    s = samples.(cn).s;
    p = samples.(cn).p;
    t = (0:cfg.N-1) * cfg.Ts * 1e6;

    subplot(4, 3, (idx-1)*3 + 1);
    wrapped_phase = angle(s);
    active = false(cfg.N,1); active(p.start_idx:p.stop_idx) = true;
    wrapped_phase(~active) = NaN;
    plot(t, wrapped_phase, 'w.', 'MarkerSize', 4);
    xlabel('t [\mus]'); ylabel('phase [rad]');
    title(sprintf('%s  (N=%d)  Wrapped phase', cn, p.N));
    ylim([-pi-0.3, pi+0.3]); grid on;

    subplot(4, 3, (idx-1)*3 + 2);
    stem(1:p.num_chips, p.phase_sequence, 'filled', 'MarkerSize', 3);
    xlabel('chip index k'); ylabel('phase [rad]');
    title(sprintf('%s phase sequence (%d chips)', cn, p.num_chips));
    ylim([-pi-0.3, pi+0.3]); grid on;

    subplot(4, 3, (idx-1)*3 + 3);
    spectrogram(s, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title([cn, '  -  STFT spectrogram']);
end

% ------------------------------------------------------------------
% Figure 3: 5-class comparison
% ------------------------------------------------------------------
figure('Name', '5-class comparison', 'Position', [180 30 1300 900]);

[s_lfm, p_lfm] = generate_lfm(cfg);

s_nlfm = []; p_nlfm = [];
for trial = 1:50
    [s_, p_] = generate_nlfm(cfg);
    if strcmp(p_.variant, 'quadratic'), s_nlfm=s_; p_nlfm=p_; break; end
end

[s_brk, p_brk] = generate_barker(cfg);
[s_frk, p_frk] = generate_frank(cfg);

% Force a P3 polyphase (looks most like LFM, good for comparison)
s_pp = []; p_pp = [];
for trial = 1:80
    [s_, p_] = generate_polyphase(cfg);
    if strcmp(p_.subcode, 'P3'), s_pp=s_; p_pp=p_; break; end
end
if isempty(s_pp), [s_pp, p_pp] = generate_polyphase(cfg); end

quins = {
    s_lfm,  p_lfm,  'LFM (linear)';
    s_nlfm, p_nlfm, 'NLFM (quadratic)';
    s_brk,  p_brk,  sprintf('Barker (%s)', p_brk.code_name);
    s_frk,  p_frk,  sprintf('Frank (N=%d)', p_frk.N);
    s_pp,   p_pp,   sprintf('Polyphase (%s, N=%d)', p_pp.subcode, p_pp.N)
};

for k = 1:5
    s = quins{k,1}; p = quins{k,2}; name = quins{k,3};

    subplot(5, 2, (k-1)*2 + 1);
    wrapped_phase = angle(s);
    active = false(cfg.N,1); active(p.start_idx:p.stop_idx) = true;
    wrapped_phase(~active) = NaN;
    plot((0:cfg.N-1)*cfg.Ts*1e6, wrapped_phase, 'w.', 'MarkerSize', 3);
    xlabel('t [\mus]'); ylabel('phase [rad]');
    title([name, '  -  Wrapped phase']);
    ylim([-pi-0.3, pi+0.3]); grid on;

    subplot(5, 2, (k-1)*2 + 2);
    spectrogram(s, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title([name, '  -  STFT spectrogram']);
end

% ------------------------------------------------------------------
% Figure 4: Polyphase at multiple SNRs
% ------------------------------------------------------------------
figure('Name', 'Polyphase at different SNRs', 'Position', [240 60 1200 700]);
snr_test = [-10, 0, 10, 20];

for i = 1:numel(snr_test)
    s_noisy = add_awgn(s_pp, snr_test(i), [p_pp.start_idx, p_pp.stop_idx]);

    subplot(numel(snr_test), 2, 2*i - 1);
    t = (0:cfg.N-1) * cfg.Ts * 1e6;
    plot(t, real(s_noisy));
    title(sprintf('Polyphase (%s, N=%d), real part, SNR = %d dB', ...
          p_pp.subcode, p_pp.N, snr_test(i)));
    xlabel('t [\mus]'); ylabel('amp'); grid on;

    subplot(numel(snr_test), 2, 2*i);
    spectrogram(s_noisy, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title(sprintf('Polyphase (%s) spectrogram, SNR = %d dB', ...
          p_pp.subcode, snr_test(i)));
end

fprintf('Tests complete. Inspect four figures.\n');
fprintf('  - Figure 1: 8 polyphase pulses; mix of all 4 subcodes expected.\n');
fprintf('  - Figure 2: P1/P2 = matrix-organized; P3/P4 = quadratic-LFM-like.\n');
fprintf('  - Figure 3: Polyphase-P3 spectrogram should resemble LFM (this is\n');
fprintf('             the classification challenge). Frank should look distinct.\n');
fprintf('  - Figure 4: Polyphase fades into noise at SNR = -10 dB.\n');
