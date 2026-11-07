// ---------------------------------------------------------------------------
// One icon + count block per running provider, at the top centre of the screen;
// the provider's weekly-limit figure joins the count as a second line.
// The window is translucent, click-through and hidden while no agents run.
// ---------------------------------------------------------------------------
#pragma once

#include <QWidget>
#include <QPixmap>

#include "Registry.h"

// A provider block shows its weekly figure under the count WHENEVER the
// provider reported one — no threshold (user decision 2026-11-06, replacing
// the first "below 25 %"). A block without a figure keeps the one-line look.
bool showsWeekLeft(const AgentGroup& group);

// The weekly line as each provider's own site states it (user, 2026-11-06):
// Codex counts the REMAINDER down ("↓76%"), Claude counts USE up
// ("↑24%" = 100 − remainder). The wire carries the remainder for both
// (`week_left`); an untyped sender gets the bare remainder, no arrow.
QString weekFigureText(const QString& agentType, int weekLeft);

class Badge : public QWidget {
	Q_OBJECT
public:
	Badge();

	// `oldestSeconds` is not decoration: the window is click-through, so there
	// is no hover and no tooltip, and colour is the only channel left to tell
	// "an agent is working" from "an agent has been stuck since this morning".
	void setState(const QVector<AgentGroup>& groups);

protected:
	void paintEvent(QPaintEvent* event) override;

private:
	void reposition();

	QVector<AgentGroup> m_groups;
	QPixmap m_claudeIcon;
	QPixmap m_codexIcon;
};
