from __future__ import annotations

from v3.training.trainer import SupervisedTrainerV3


def run_plain(
    args,
    model_class,
    *,
    trainer_class=SupervisedTrainerV3,
    train_range=None,
    valid_range=None,
    test_range=None,
    prediction_range=None,
    run_name=None,
):
    trainer = trainer_class(args, model_class, run_name=run_name)
    history = trainer.fit(train_range=train_range, valid_range=valid_range)
    pred, label = trainer.predict(prediction_range or test_range, save=True)
    return pred, label, [history]
