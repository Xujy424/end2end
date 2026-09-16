from __future__ import annotations


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
):
    trainer = trainer_class(args, model_class)
    history = trainer.fit(train_range=train_range, valid_range=valid_range)
    pred, label = trainer.predict(prediction_range or test_range, save=True)
    return pred, label, [history]
