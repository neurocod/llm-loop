#include "Badge.h"

#include <QFontMetrics>
#include <QGuiApplication>
#include <QPainter>
#include <QScreen>

namespace {

constexpr int kHeight = 30;      // logical px; Qt scales it per monitor DPI
// 22 px icon, gap and a 36 px text column: room for three count glyphs, and
// for the week line ("↑76%": arrow, two digits and "%") at nearly its full
// 11 px ink height — at 60 px (27 px column) that line shrank to ~8 px (2026-11-06).
constexpr int kBlockWidth = 68;
constexpr int kColumnLeft = 28;  // from the block's left; the icon ends at 27
constexpr int kColumnWidth = 36; // ends 4 px inside the 8 px rounded corner
constexpr int kGap = 4;
constexpr int kFillAlpha = 190;   // enough to read white digits, still see-through

// The count's pixel size when it is the block's only line.
constexpr int kCountPixels = 18;

// Two lines (count over weekly remainder) are laid out by INK, not by the
// font's line box: ascent and descent keep room for accents and descenders
// (the tail of a "q") that digits, "+" and "%" never use, and a 9 px font
// centred in half the block left most of the block empty (user, 2026-11-06).
// Each line gets an 11 px tall box of ink: 3 px margin, 11, a 2 px gap, 11,
// 3 px margin = the 30 px block. BadgeTest's isTwoLine reads the gap rows.
constexpr int kTwoLineMargin = 3;
constexpr int kTwoLineGap = 2;
constexpr int kTwoLineInk = (kHeight - 2 * kTwoLineMargin - kTwoLineGap) / 2;

// The largest pixel size whose ink for `text` fits `height` x `width`; `font`
// comes back set to it. Width shrinks only the line that needs it.
QRect fitInk(QFont& font, const QString& text, int height, int width) {
	QRect ink;
	for (int pixels = 2 * height; pixels > 4; --pixels) {
		font.setPixelSize(pixels);
		ink = QFontMetrics(font).tightBoundingRect(text);
		if (ink.height() <= height && ink.width() <= width)
			break;
	}
	return ink;
}

// The week line's leading arrow is PAINTED, not typed: the font's U+2191/2193
// is a thin glyph that reaches below the baseline, so as text it both looked
// unlike the bold digits and, being taller than them, shrank the whole line
// (measured 2026-11-06 on the badge preview). Drawn at the digits' own ink
// height with the digits' stroke, it matches them.
constexpr double kArrowWidthPerHeight = 0.6;
constexpr int kArrowGap = 1;

void drawArrow(QPainter& painter, const QRectF& ink, bool up) {
	const double x = ink.center().x();
	const double head = ink.width() * 0.9;
	const double tip = up ? ink.top() : ink.bottom();
	const double tail = up ? ink.bottom() : ink.top();
	const double base = up ? tip + head : tip - head;
	painter.save();
	painter.setPen(QPen(Qt::white, qMax(1.5, ink.width() * 0.3), Qt::SolidLine, Qt::FlatCap));
	painter.drawLine(QPointF(x, tail), QPointF(x, base));
	painter.setPen(Qt::NoPen);
	painter.setBrush(Qt::white);
	const QPointF triangle[3] = {QPointF(x, tip), QPointF(ink.left(), base), QPointF(ink.right(), base)};
	painter.drawPolygon(triangle, 3);
	painter.restore();
}

// One line laid out by ink in `box`, centred on that ink: ascent and descent
// are not part of it (see kTwoLineMargin). A leading arrow from
// weekFigureText is drawn by drawArrow beside the fitted digits.
void drawInkFitted(QPainter& painter, QFont font, const QRect& box, const QString& text) {
	const bool up = text.startsWith(QChar(0x2191));
	const bool arrow = up || text.startsWith(QChar(0x2193));
	const QString digits = arrow ? text.mid(1) : text;
	const int arrowWidth = arrow ? qRound(box.height() * kArrowWidthPerHeight) : 0;
	const int lead = arrow ? arrowWidth + kArrowGap : 0;
	const QRect ink = fitInk(font, digits, box.height(), box.width() - lead);
	painter.setFont(font);
	// The arrow keeps the digits' height even when width shrank them, so its
	// width follows the fitted ink, not the box (`lead` only bounded the fit).
	const double arrowDrawn = arrow ? ink.height() * kArrowWidthPerHeight + kArrowGap : 0.0;
	const double left = box.left() + (box.width() - arrowDrawn - ink.width()) / 2.0;
	const double top = box.top() + (box.height() - ink.height()) / 2.0;
	if (arrow)
		drawArrow(painter, QRectF(left, top, arrowDrawn - kArrowGap, ink.height()), up);
	// tightBoundingRect is relative to the baseline origin, so its left/top
	// offsets are subtracted to put the ink, not the origin, where it belongs.
	painter.drawText(QPointF(left + arrowDrawn - ink.left(), top - ink.top()), digits);
}

// Age thresholds behind each provider block's colour. A turn of a couple of minutes
// is ordinary; ten minutes is either a very large review or something wedged,
// and either way it is what the operator wants to notice from across the room.
constexpr int kAmberSeconds = 120;
constexpr int kRedSeconds = 600;

QColor ageColour(int seconds) {
	if (seconds >= kRedSeconds)
		return QColor(198, 40, 40);
	if (seconds >= kAmberSeconds)
		return QColor(249, 168, 37);
	return QColor(46, 125, 50);
}

// The mark painted when the build embedded no provider icon (CMakeLists.txt,
// AGENT_COUNTER_ICON_DIR): a disc with a letter, original to this project so
// that it ships under the project's own licence. Painted at 3x the 22 px icon
// box, so the downscale in paintEvent stays smooth on a high-DPI monitor.
// The two marks differ in colour AND letter: the block behind them is already
// coloured by age, so a letter alone would be the only cue on a red block.
QPixmap neutralIcon(QChar letter, const QColor& disc, const QColor& ink) {
	constexpr int kSide = 66;
	QPixmap pixmap(kSide, kSide);
	pixmap.fill(Qt::transparent);
	QPainter painter(&pixmap);
	painter.setRenderHint(QPainter::Antialiasing);
	painter.setPen(Qt::NoPen);
	painter.setBrush(disc);
	painter.drawEllipse(QRectF(1, 1, kSide - 2, kSide - 2));
	QFont font;
	font.setBold(true);
	font.setPixelSize(kSide * 6 / 10);
	painter.setFont(font);
	painter.setPen(ink);
	painter.drawText(QRect(0, 0, kSide, kSide), Qt::AlignCenter, QString(letter));
	return pixmap;
}

// The build's embedded icon if there is one, else the neutral mark.
QPixmap providerIcon(const QString& resource, QChar letter, const QColor& disc, const QColor& ink) {
	QPixmap embedded(resource);
	return embedded.isNull() ? neutralIcon(letter, disc, ink) : embedded;
}

}  // namespace

bool showsWeekLeft(const AgentGroup& group) {
	return group.weekLeft.has_value();
}

QString weekFigureText(const QString& agentType, int weekLeft) {
	if (agentType == QLatin1String("claude"))
		return QStringLiteral("↑%1%").arg(100 - weekLeft);
	if (agentType == QLatin1String("codex"))
		return QStringLiteral("↓%1%").arg(weekLeft);
	return QStringLiteral("%1%").arg(weekLeft);
}

Badge::Badge()
    : QWidget(nullptr,
              Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint | Qt::Tool
                  | Qt::WindowDoesNotAcceptFocus | Qt::WindowTransparentForInput),
      m_claudeIcon(providerIcon(QStringLiteral(":/agents/claude.png"), QLatin1Char('C'),
                                QColor(245, 240, 232), QColor(176, 84, 52))),
      m_codexIcon(providerIcon(QStringLiteral(":/agents/codex.png"), QLatin1Char('X'),
                               QColor(24, 24, 24), Qt::white)) {
	// Three attributes, three separate ways this window would otherwise be a
	// nuisance rather than a hint:
	//   WA_TranslucentBackground  - without it the "transparent" corners paint
	//                               the theme's window colour as a grey square;
	//   WA_ShowWithoutActivating  - without it the badge STEALS FOCUS at the
	//                               moment an agent starts, which is exactly
	//                               when the operator is typing something else;
	//   WindowTransparentForInput - the top centre of the screen is the browser
	//                               tab strip and the title bar of every
	//                               maximised window, so clicks must pass
	//                               through (see Badge.h on what this costs).
	setAttribute(Qt::WA_TranslucentBackground);
	setAttribute(Qt::WA_ShowWithoutActivating);
	setFixedSize(kBlockWidth, kHeight);
	reposition();
}

void Badge::setState(const QVector<AgentGroup>& groups) {
	m_groups.clear();
	for (const AgentGroup& group : groups)
		if (group.count > 0)
			m_groups.append(group);
	if (m_groups.isEmpty()) {
		hide();
		return;
	}
	setFixedSize(static_cast<int>(m_groups.size()) * (kBlockWidth + kGap) - kGap, kHeight);
	// Re-aimed on every change rather than wired to screen signals: the badge
	// only matters while it is visible, and this covers a resolution change, a
	// taskbar that moved and a different primary monitor with one call.
	reposition();
	show();
	update();
}

void Badge::reposition() {
	const QScreen* screen = QGuiApplication::primaryScreen();
	if (!screen)
		return;
	// availableGeometry, not geometry: on a desktop whose taskbar sits at the
	// top, geometry would put the badge underneath it.
	const QRect area = screen->availableGeometry();
	move(area.x() + (area.width() - width()) / 2, area.y());
}

void Badge::paintEvent(QPaintEvent*) {
	QPainter painter(this);
	painter.setRenderHint(QPainter::Antialiasing);
	painter.setRenderHint(QPainter::SmoothPixmapTransform);
	int x = 0;
	for (const AgentGroup& group : m_groups) {
		const QRect block(x, 0, kBlockWidth, kHeight);
		QColor fill = ageColour(group.oldestSeconds);
		fill.setAlpha(kFillAlpha);
		painter.setBrush(fill);
		painter.setPen(Qt::NoPen);
		painter.drawRoundedRect(block.adjusted(1, 1, -1, -1), 8, 8);

		const QRect iconRect(x + 5, 4, 22, 22);
		const QPixmap* icon = nullptr;
		if (group.agentType == QLatin1String("claude"))
			icon = &m_claudeIcon;
		else if (group.agentType == QLatin1String("codex"))
			icon = &m_codexIcon;
		QFont numerals = font();
		numerals.setBold(true);
		numerals.setPixelSize(kCountPixels);
		painter.setFont(numerals);
		painter.setPen(Qt::white);
		if (icon && !icon->isNull())
			painter.drawPixmap(iconRect, *icon);
		else
			// Older senders without type metadata still count; never brand them
			// as one provider based on a caller-controlled label.
			painter.drawText(iconRect, Qt::AlignCenter, QStringLiteral("?"));

		const QString text = group.count > 99 ? QStringLiteral("99+") : QString::number(group.count);
		const QRect column(x + kColumnLeft, 0, kColumnWidth, kHeight);
		if (showsWeekLeft(group)) {
			// Count on top, remainder below, each centred in its half; the
			// block keeps its width so a provider crossing the threshold
			// does not shift its neighbour.
			const QRect top(column.left(), kTwoLineMargin, column.width(), kTwoLineInk);
			drawInkFitted(painter, numerals, top, text);
			drawInkFitted(painter, numerals, top.translated(0, kTwoLineInk + kTwoLineGap),
			              weekFigureText(group.agentType, *group.weekLeft));
		} else {
			numerals.setPixelSize(text.size() > 2 ? 13 : kCountPixels);
			painter.setFont(numerals);
			painter.drawText(column, Qt::AlignCenter, text);
		}
		x += kBlockWidth + kGap;
	}
}
