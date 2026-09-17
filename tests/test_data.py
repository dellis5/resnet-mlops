from src.data.dataset import build_transforms


def test_build_transforms_train_vs_eval_differ():
    train_t = build_transforms(224, train=True)
    eval_t = build_transforms(224, train=False)
    assert len(train_t.transforms) != len(eval_t.transforms) or type(train_t.transforms[0]) != type(
        eval_t.transforms[0]
    )
