# 用途：独立检查配置、物理单位、同步能耗与切点时序；不运行正式训练。
# 运行指令（在 code 目录）：python -B -m unittest discover -s tests -v
# 手算例子使用小整数方便检查公式；这些数值不是实际 UAV 参数。
from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment_config as ec
import resource_allocation as ra
import split_selection as ss
from data import CLASSES, _fingerprint
from training_common import common_parser


def config_fixture():
    cfg = ec.load_config()
    cfg['uavs'] = cfg['uavs'][:2]
    for i, u in enumerate(cfg['uavs']):
        u.update(f_min_hz=1., f_max_hz=5., f_ref_hz=[2., 4.][i],
                 kappa=[.1, .2][i], cycles_per_flop=1., p_up_min_w=1.,
                 p_up_max_w=20., p_up_ref_w=10., hover_power_w=[10., 20.][i])
    cfg['server'].update(parallel_branches=2, frequency_hz=4., cycles_per_flop=1.,
                         kappa=.2, downlink_power_w=10., aggregation_time_s=.5,
                         aggregation_power_w=4.)
    cfg['radio'].update(total_up_hz=20., total_down_hz=20., min_up_hz=1., min_down_hz=1.)
    cfg['environment']['noise_psd_w_hz'] = 1.
    cfg['selection'].update(candidates=['P0', 'P1'], initial_split='P0')
    ec.validate_config(cfg)
    return cfg


def profiles_fixture():
    return {
        'P0': dict(client_flops_per_sample=6., server_flops_per_sample=8.,
                   activation_bytes_per_sample=1., gradient_bytes_per_sample=1.,
                   label_bytes_per_sample=1., prefix_state_bytes=4.),
        'P1': dict(client_flops_per_sample=12., server_flops_per_sample=4.,
                   activation_bytes_per_sample=2., gradient_bytes_per_sample=2.,
                   label_bytes_per_sample=1., prefix_state_bytes=8.),
    }


def manifest_fixture():
    rows = []
    for i, split in enumerate(['train'] * 5 + ['val', 'test']):
        rows.append(dict(sample_id=str(i), scene_id=split + str(i), split=split,
                         label=i % 9, class_name=CLASSES[i % 9], crop=f'crops/{i}.png'))
    manifest = dict(schema=1, settings={'uavs': 2}, classes=list(CLASSES), samples=rows,
                    clients=[dict(indices=[0, 1], samples=2), dict(indices=[2, 3, 4], samples=3)],
                    subset_counts={'train': 5, 'val': 1, 'test': 1})
    manifest['fingerprint'] = _fingerprint(manifest)
    return manifest


class ConfigTests(unittest.TestCase):
    def test_defaults_copy_and_bad_hardware(self):
        cfg = ec.load_config()
        ec.validate_config(cfg)
        cfg['training']['rounds'] = 7
        self.assertEqual(ec.load_config()['training']['rounds'], 100)
        cfg['server']['parallel_branches'] = 4
        with self.assertRaises(ValueError):
            ec.validate_config(cfg)
        cfg = ec.load_config()
        cfg['uavs'][0]['f_ref_hz'] = float('nan')
        with self.assertRaises(ValueError):
            ec.validate_config(cfg)

    def test_five_parsers_and_explicit_overrides(self):
        for method in ['centralized', 'local', 'fedavg', 'split_fixed', 'split_dynamic']:
            with self.subTest(method=method):
                args = common_parser('test', method).parse_args([])
                self.assertEqual((args.rounds, args.batch_size, args.lr, args.bn), (100, 32, .01, 'batch'))
                self.assertTrue(Path(args.data).as_posix().endswith('output/data_load_7500/data.json'))
        with patch.dict(ec.CONFIG['training'], rounds=7, batch_size=5, lr=.02):
            parser = common_parser('test', 'fedavg')
            self.assertEqual(parser.parse_args([]).rounds, 7)
            args = parser.parse_args(['--rounds', '3', '--batch-size', '2', '--lr', '.03'])
            self.assertEqual((args.rounds, args.batch_size, args.lr), (3, 2, .03))
            snap = ec.configuration_snapshot(args, 'fedavg')
            self.assertEqual(snap['resolved']['training']['rounds'], 3)
            self.assertFalse(snap['resource_optimization_applied'])
            json.dumps(snap, allow_nan=False)
        full = ec.training_defaults('resnet18_full')
        self.assertTrue(Path(full['data']).as_posix().endswith('output/data_full/data.json'))
        self.assertEqual(full['epochs'], 100)

    def test_manifest_without_images_and_mismatch(self):
        cfg, manifest = config_fixture(), manifest_fixture()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'data.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            workload = ra.read_manifest_workload(path, cfg)
            self.assertEqual(workload['sizes'], [2, 3])
            self.assertEqual(len(workload['manifest_sha256']), 64)
            with self.assertRaises(ValueError):
                ra.read_manifest_workload(path, ec.load_config())
            manifest['clients'][1]['indices'] = [1, 3, 4]
            manifest['fingerprint'] = _fingerprint(manifest)
            path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                ra.read_manifest_workload(path, cfg)


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.profiles = config_fixture(), profiles_fixture()
        self.resources = ra.reference_resources(self.cfg)

    def problem(self, **kw):
        return ra.build_round_problem(self.cfg, self.profiles, [2, 3], [1., 1.], ['P0', 'P1'], **kw)

    def test_units_stages_hover_and_energy_hand_calculation(self):
        problem = self.problem(next_cuts=['P1', 'P0'], aggregation_due=True)
        self.assertEqual(problem['client_cycles'], [12., 36.])
        self.assertEqual(problem['payload_bits']['activation_up'], [32., 72.])
        self.assertEqual(problem['payload_bits']['gradient_down'], [16., 48.])
        self.assertEqual(problem['payload_bits']['model_down'], [64., 32.])
        result = ra.evaluate_resources(problem, self.resources)
        self.assertAlmostEqual(result['system_time_s'], 37.3)
        self.assertAlmostEqual(result['system_energy_j'], 1690.6)
        for actual, expected in zip(result['uav_energy_j'], [441.8, 997.2]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result['times_s']['stages'], dict(sl=24., model_up=6.4, model_down=6.4, aggregation=.5))
        self.assertEqual(result['uav_compute_power_w'], [.8, 12.8])
        self.assertTrue(result['feasible'])
        self.assertFalse(result['optimized'])
        self.assertFalse(result['energy_measured'])

    def test_prefix_upload_not_multiplied_by_samples_and_final_no_down(self):
        result = self.problem(next_cuts=['P1', 'P0'], aggregation_due=True, final_round=True)
        self.assertEqual(result['payload_bits']['model_up'], [32., 64.])
        self.assertEqual(result['payload_bits']['model_down'], [0., 0.])
        plain = self.problem()
        self.assertEqual(plain['payload_bits']['model_up'], [0., 0.])
        self.assertEqual(plain['payload_bits']['model_down'], [0., 0.])

    def test_shared_bandwidth_frequency_and_round_budgets(self):
        bad = copy.deepcopy(self.resources)
        bad['bandwidth_up_hz'] = [15., 15.]
        self.assertFalse(ra.validate_resources(self.problem(), bad)['feasible'])
        bad = copy.deepcopy(self.resources)
        bad['f_hz'][0] = 6.
        self.assertFalse(ra.evaluate_resources(self.problem(), bad)['feasible'])
        self.cfg['budgets']['round_time_s'] = 1.
        self.assertFalse(ra.validate_resources(self.problem(), self.resources)['feasible'])
        self.cfg['budgets']['round_time_s'] = None
        self.cfg['budgets']['round_uav_energy_j'] = [1., 1.]
        self.assertFalse(ra.evaluate_resources(self.problem(), self.resources)['feasible'])
        bad['f_hz'][0] = -1.
        with self.assertRaises(ValueError):
            ra.evaluate_resources(self.problem(), bad)

    def test_cycle_preload_and_single_epoch_budget(self):
        args = (self.cfg, self.profiles, [2, 3], [1., 1.], ['P0', 'P0'], self.resources, 2)
        loaded = ra.evaluate_cycle_reference(*args, preloaded=True)
        normal = ra.evaluate_cycle_reference(*args, preloaded=False)
        self.assertAlmostEqual(loaded['cycle_time_s'], 39.1)
        self.assertAlmostEqual(normal['cycle_time_s'], 42.3)
        self.assertAlmostEqual(loaded['cycle_energy_j'], 1859.8)
        self.assertAlmostEqual(normal['cycle_energy_j'], 2019.8)
        self.cfg['budgets']['round_time_s'] = 23.
        result = ra.evaluate_cycle_reference(self.cfg, self.profiles, [2, 3], [1., 1.],
                                             ['P0', 'P0'], self.resources, 1, preloaded=False)
        self.assertTrue(result['feasible'])  # 下发归上一轮；下一轮 SL+上传+聚合为 21.4s。
        self.cfg['budgets']['round_time_s'] = 20.
        result = ra.evaluate_cycle_reference(self.cfg, self.profiles, [2, 3], [1., 1.],
                                             ['P0', 'P0'], self.resources, 1, preloaded=True)
        self.assertFalse(result['feasible'])  # SL 单独可行，但同一轮上传/聚合后超限。

    def test_public_resource_apis_reject_invalid_config(self):
        self.cfg['uavs'][0]['kappa'] = -1.
        with self.assertRaises(ValueError):
            self.problem()
        with self.assertRaises(ValueError):
            ra.reference_resources(self.cfg)
        with self.assertRaises(ValueError):
            ra.observe_channel(self.cfg, 0)

    def test_start_download_scored_once_and_budgeted_in_transition(self):
        self.cfg['budgets']['round_time_s'] = 23.
        p = self.profiles
        # 当前前缀很小且 SL 负载很轻；候选 P0 下发在这一轮末完成。
        p['P1'].update(client_flops_per_sample=0., server_flops_per_sample=0.,
                       activation_bytes_per_sample=0., gradient_bytes_per_sample=0.,
                       label_bytes_per_sample=0., prefix_state_bytes=0.)
        # 单独构造当前两架都是 P1 的实际过渡轮，再检查下一最终轮。
        current = ra.build_round_problem(self.cfg, p, [2, 3], [1., 1.], ['P1', 'P1'],
                                         next_cuts=['P0', 'P0'], aggregation_due=True)
        transition = ra.evaluate_resources(current, self.resources)
        next_problem = ra.build_round_problem(self.cfg, p, [2, 3], [1., 1.], ['P0', 'P0'],
                                              aggregation_due=True, final_round=True)
        future = ra.evaluate_resources(next_problem, self.resources)
        scored = ra.evaluate_cycle_reference(self.cfg, p, [2, 3], [1., 1.],
                                              ['P0', 'P0'], self.resources, 1, preloaded=False)
        self.assertTrue(transition['feasible'])
        self.assertTrue(future['feasible'])
        self.assertTrue(scored['feasible'])
        self.assertAlmostEqual(scored['cycle_time_s'], 24.6)
        self.assertAlmostEqual(future['system_time_s'], 21.4)
        self.assertAlmostEqual(transition['system_time_s'], 3.7)

    def test_channel_reproducible_los_fixed_and_fading_changes(self):
        cfg = ec.load_config()
        a, b = ra.observe_channel(cfg, 0), ra.observe_channel(cfg, 1)
        self.assertEqual(a, ra.observe_channel(cfg, 0))
        self.assertEqual(a['los'], b['los'])
        self.assertNotEqual(a['gain'], b['gain'])
        self.assertTrue(all(g > 0 and math.isfinite(g) for g in a['gain']))
        with self.assertRaises(ValueError):
            ra.observe_channel(cfg, -1)

    def test_low_snr_and_solver_placeholder(self):
        result = ra.evaluate_resources(ra.build_round_problem(self.cfg, self.profiles, [2, 3],
                                                               [1e-20, 1e-20], ['P0', 'P0']), self.resources)
        self.assertTrue(math.isfinite(result['system_time_s']))
        with self.assertRaisesRegex(NotImplementedError, '尚未实现'):
            ra.allocate_resources(self.problem())

    def test_cumulative_mission_budget_and_json_output(self):
        row = ra.evaluate_resources(self.problem(), self.resources)
        self.cfg['budgets']['mission_time_s'] = row['system_time_s'] * 1.5
        workload = dict(sizes=[2, 3], manifest_sha256='fixture')
        with tempfile.TemporaryDirectory() as temp:
            summary = ra.save_reference_plan(temp, self.cfg, workload, self.profiles,
                                              [row, row], [], label='test_reference')
            self.assertFalse(summary['feasible'])
            self.assertFalse(summary['optimized'])
            self.assertEqual(summary['rounds'], 2)
            for path in Path(temp).glob('*.json'):
                json.loads(path.read_text(encoding='utf-8'), parse_constant=lambda v: self.fail(v))


class SelectionTests(unittest.TestCase):
    def test_legacy_latency_numeric_snapshot_and_tie_order(self):
        p = profiles_fixture()
        cuts, cost = ss.choose_latency_cuts(p, [3, 4], [2., 1.], [20., 10.], 100., 2)
        self.assertEqual(cuts, ['P0', 'P0'])
        self.assertAlmostEqual(cost[0]['P0'], 5.20924e-6, places=13)
        self.assertAlmostEqual(cost[0]['P1'], 9.21812e-6, places=13)
        self.assertAlmostEqual(cost[1]['P0'], 12.82432e-6, places=13)
        same = {'P1': p['P0'], 'P0': p['P0']}
        self.assertEqual(ss.choose_latency_cuts(same, [3], [2.], [20.], 100., 1)[0], ['P1'])

    def test_tradeoff_tie_keeps_previous_and_infeasible_candidates(self):
        cfg, p = config_fixture(), profiles_fixture()
        p['P1'] = copy.deepcopy(p['P0'])
        resources = ra.reference_resources(cfg)
        result = ss.select_tradeoff_cuts(cfg, p, [2, 3], [1., 1.], resources, ['P1', 'P1'], 2)
        self.assertEqual(result['cuts'], ['P1', 'P1'])
        self.assertEqual(result['audit'][0]['delta_energy_j'], 0.)
        json.dumps(result, allow_nan=False)
        cfg['budgets']['round_time_s'] = .001
        with self.assertRaises(ValueError):
            ss.select_tradeoff_cuts(cfg, p, [2, 3], [1., 1.], resources, ['P1', 'P1'], 2)

    def load_tradeoff_fixture(self):
        cfg = config_fixture()
        cfg['radio'].update(total_up_hz=2., total_down_hz=2., min_up_hz=.1, min_down_hz=.1)
        cfg['environment']['noise_psd_w_hz'] = 1e-5
        cfg['server'].update(downlink_power_w=1e-5, aggregation_time_s=0.)
        for u in cfg['uavs']:
            u.update(f_min_hz=.5, f_max_hz=2., f_ref_hz=1., kappa=1., hover_power_w=1.,
                     p_up_min_w=1e-6, p_up_max_w=1., p_up_ref_w=1e-5)
        p = {cut: dict(client_flops_per_sample=c, server_flops_per_sample=0.,
                       activation_bytes_per_sample=a, gradient_bytes_per_sample=0.,
                       label_bytes_per_sample=0., prefix_state_bytes=0.)
             for cut, c, a in [('P0', 1., 1.), ('P1', 2., 0.)]}
        ec.validate_config(cfg)
        return cfg, p

    def test_global_hover_is_coupled_not_per_uav_latency(self):
        cfg, p = self.load_tradeoff_fixture()
        result = ss.select_tradeoff_cuts(cfg, p, [1, 100], [1., 1.], ra.reference_resources(cfg), ['P0', 'P0'], 1, preloaded=True)
        self.assertEqual(result['cuts'], ['P0', 'P1'])
        # UAV0 换点虽更快，但 UAV1 仍是瓶颈，不能降低全体悬停；额外计算使它保留 P0。
        self.assertEqual(result['audit'][0]['selected_cut'], 'P0')
        self.assertGreater(result['audit'][0]['candidates'][1]['delta_energy_j'], 0.)

    def test_next_prefix_transition_budget_excludes_candidate(self):
        cfg, p = self.load_tradeoff_fixture()
        cfg['uavs'] = cfg['uavs'][:1]
        cfg['budgets']['round_time_s'] = 16.
        cfg['server']['aggregation_time_s'] = .5
        p['P0'].update(client_flops_per_sample=10., activation_bytes_per_sample=0.)
        p['P1'].update(client_flops_per_sample=1., prefix_state_bytes=1.)
        resources = ra.reference_resources(cfg)
        selected = ss.select_tradeoff_cuts(cfg, p, [1], [1.], resources, ['P0'], 1)
        self.assertEqual(selected['cuts'], ['P1'])
        guarded = ss.select_tradeoff_cuts(cfg, p, [1], [1.], resources, ['P0'], 1, transition_cuts=['P0'])
        self.assertEqual(guarded['cuts'], ['P0'])
        self.assertTrue(guarded['audit'][0]['candidates'][1]['transition_violations'])

    def test_rounds_cycle_boundaries_no_future_channel_and_final_down(self):
        cfg, p = config_fixture(), profiles_fixture()
        cfg['training'].update(rounds=5, aggregation_period=2)
        p['P1'] = copy.deepcopy(p['P0'])
        def observe(config, index):
            return dict(round_index=index, gain=[1., 1.])
        with patch.object(ra, 'observe_channel', side_effect=observe) as observer:
            history, schedule = ss.run_reference_plan(cfg, {'sizes': [2, 3]}, p)
        self.assertEqual([c.args[1] for c in observer.call_args_list], list(range(5)))
        self.assertEqual([s['effective_round'] for s in schedule], [1, 3, 5])
        self.assertEqual([s['period'] for s in schedule], [2, 2, 1])
        self.assertEqual(history[0]['payload_bits']['model_up'], [0., 0.])
        self.assertEqual(history[-1]['payload_bits']['model_down'], [0., 0.])
        self.assertGreater(sum(history[-1]['payload_bits']['model_up']), 0.)
        self.assertEqual(sum(sum(r['payload_bits']['model_up']) for r in history), 192.)
        self.assertEqual(sum(sum(r['payload_bits']['model_down']) for r in history), 128.)
        for i, row in enumerate(history[:-1]):
            if row['next_cuts'] is not None:
                self.assertEqual(row['next_cuts'], history[i+1]['cuts'])
        json.dumps({'history': history, 'schedule': schedule}, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
