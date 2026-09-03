"""Resumable nnU-Net v2 preprocessing (stage 3).

Blocks the rmtree that wipes the output folder and drops cases that are
already on disk. A case counts as done only if its .pkl exists, because
nnU-Net writes that after the arrays, so partial cases get redone.
"""

import argparse
import inspect
import os
import shutil as _real_shutil
from os.path import isfile, isdir, join, exists

DATA_SUFFIXES = ('.npz', '.npy', '.b2nd', '_seg.npy', '_seg.b2nd')


class _ShutilProxy:
    def __init__(self, real):
        self._real = real

    def rmtree(self, path, *args, **kwargs):
        print('[resume] blocked rmtree of ' + str(path))

    def __getattr__(self, name):
        return getattr(self._real, name)


def case_is_complete(output_dir, identifier):
    if not isfile(join(output_dir, identifier + '.pkl')):
        return False
    for suffix in DATA_SUFFIXES:
        p = join(output_dir, identifier + suffix)
        if exists(p) and (not isfile(p) or os.path.getsize(p) > 0):
            return True
    return False


def split_done(output_dir, identifiers):
    if not isdir(output_dir):
        return [], list(identifiers)
    done = [i for i in identifiers if case_is_complete(output_dir, i)]
    todo = [i for i in identifiers if i not in set(done)]
    return done, todo


def install_patches(module, output_dir, originals, force=False):
    if hasattr(module, 'shutil'):
        module.shutil = _ShutilProxy(_real_shutil)
    if hasattr(module, 'rmtree'):
        module.rmtree = lambda path, *a, **kw: print('[resume] blocked rmtree')

    for name in ('get_filenames_of_train_images_and_targets',
                 'get_identifiers_from_splitted_dataset_folder'):
        if not hasattr(module, name):
            continue
        if name not in originals:
            originals[name] = getattr(module, name)
        orig = originals[name]

        def make_patched(orig_fn):
            def patched(*args, **kwargs):
                result = orig_fn(*args, **kwargs)
                if force:
                    return result
                if isinstance(result, dict):
                    done, todo = split_done(output_dir, list(result.keys()))
                    print('[resume] %d done, %d to go' % (len(done), len(todo)))
                    return dict((k, result[k]) for k in todo)
                done, todo = split_done(output_dir, list(result))
                print('[resume] %d done, %d to go' % (len(done), len(todo)))
                return todo
            return patched

        setattr(module, name, make_patched(orig))


def main():
    p = argparse.ArgumentParser(description='Resumable nnU-Net v2 preprocessing.')
    p.add_argument('-d', type=int, required=True)
    p.add_argument('-c', nargs='+', default=['3d_fullres'])
    p.add_argument('-np', type=int, nargs='+', default=[2])
    p.add_argument('-plans_identifier', type=str, default='nnUNetPlans')
    p.add_argument('--force', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess
    import nnunetv2.preprocessing.preprocessors.default_preprocessor as dp

    dataset_name = maybe_convert_to_dataset_name(args.d)
    preprocessed_dir = join(nnUNet_preprocessed, dataset_name)
    plans_file = join(preprocessed_dir, args.plans_identifier + '.json')
    if not isfile(plans_file):
        raise FileNotFoundError(plans_file + ' not found. Run nnUNetv2_plan_experiment first.')
    plans = load_json(plans_file)

    n_proc = args.np if len(args.np) == len(args.c) else [args.np[0]] * len(args.c)
    originals = {}

    for cfg, npr in zip(args.c, n_proc):
        if cfg not in plans['configurations']:
            print('[resume] no configuration ' + cfg + ' in plans, skipping')
            continue
        cfg_plan = plans['configurations'][cfg]
        out_dir = join(preprocessed_dir, cfg_plan['data_identifier'])
        n_done = len([f for f in (os.listdir(out_dir) if isdir(out_dir) else [])
                      if f.endswith('.pkl')])
        print('')
        print('=== ' + cfg + ' -> ' + out_dir)
        print('[resume] %d completed case(s) currently on disk' % n_done)

        if args.dry_run:
            continue

        install_patches(dp, out_dir, originals, force=args.force)
        call_kwargs = {}
        if 'show_progress_bar' in inspect.signature(preprocess).parameters:
            call_kwargs['show_progress_bar'] = True
        preprocess([args.d], args.plans_identifier, (cfg,), (npr,), args.verbose,
                   **call_kwargs)

    print('')
    print('[resume] done')


if __name__ == '__main__':
    main()
