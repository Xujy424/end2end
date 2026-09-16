from __future__ import annotations

from v3.training.trainer import resolve_trainer_class


def run_plain(
    args,
    model_class,
    *,
    trainer="supervised",
    trainer_class=None,
    train_range=None,
    valid_range=None,
    test_range=None,
    prediction_range=None,
    run_name=None,
):
    trainer_class = resolve_trainer_class(trainer, trainer_class)
    trainer = trainer_class(args, model_class, run_name=run_name)
    history = trainer.fit(train_range=train_range, valid_range=valid_range)
    pred, label = trainer.predict(prediction_range or test_range, save=True)
    return pred, label, [history]
