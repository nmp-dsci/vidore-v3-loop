import type { RouteObject } from 'react-router-dom';
import { Shell } from './Shell';
import { Agent, AgentQuestionPage, AgentRunPage } from './pages/Agent';
import { Overview } from './pages/Overview';
import { Dataset, Datasets } from './pages/Datasets';
import { Leaderboard } from './pages/Leaderboard';
import { Pipeline } from './pages/Pipeline';
import { Review, ReviewQuestion, ReviewRun } from './pages/Review';
import { Run, RunQuestion, Runs } from './pages/Runs';

/**
 * Every viewer address, as data. The grammar follows tau2-loop's: one id per thing,
 * the path names the subject, the query string holds the lens (here, the list's
 * filters), and a detail opens beside its list.
 */
export const routes: RouteObject[] = [
  {
    id: 'shell',
    element: <Shell />,
    children: [
      { id: 'overview', path: '/', element: <Overview /> },
      { id: 'datasets', path: '/datasets', element: <Datasets /> },
      { id: 'dataset', path: '/datasets/:key', element: <Dataset /> },
      { id: 'question', path: '/datasets/:key/:qid', element: <Dataset /> },
      { id: 'pipeline', path: '/pipeline', element: <Pipeline /> },
      { id: 'runs', path: '/runs', element: <Runs /> },
      { id: 'run', path: '/runs/:runId', element: <Run /> },
      { id: 'run-question', path: '/runs/:runId/:qid', element: <RunQuestion /> },
      { id: 'review', path: '/review', element: <Review /> },
      { id: 'review-run', path: '/review/:runId', element: <ReviewRun /> },
      { id: 'review-question', path: '/review/:runId/:qid', element: <ReviewQuestion /> },
      { id: 'agent', path: '/agent', element: <Agent /> },
      { id: 'agent-run', path: '/agent/:runId', element: <AgentRunPage /> },
      { id: 'agent-question', path: '/agent/:runId/:qid', element: <AgentQuestionPage /> },
      { id: 'leaderboard', path: '/leaderboard', element: <Leaderboard /> },
      { id: 'missing', path: '*', element: <p className="empty">No page here. Try the Datasets tab.</p> },
    ],
  },
];
