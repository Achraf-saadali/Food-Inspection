/**
 * Root application with routing.
 * Design: Industrial Precision — dark theme, persistent sidebar layout
 */

import { Toaster } from '@/components/ui/sonner';
import { TooltipProvider } from '@/components/ui/tooltip';
import { Route, Switch } from 'wouter';
import ErrorBoundary from './components/ErrorBoundary';
import { ThemeProvider } from './contexts/ThemeContext';
import { AppLayout } from './components/layout/AppLayout';
import Dashboard from './pages/Dashboard';
import LiveInspection from './pages/LiveInspection';
import Reports from './pages/Reports';
import ModelInfo from './pages/ModelInfo';
import NotFound from './pages/NotFound';

function Router() {
  return (
    <AppLayout>
      <Switch>
        <Route path="/" component={Dashboard} />
        <Route path="/inspection" component={LiveInspection} />
        <Route path="/reports" component={Reports} />
        <Route path="/model" component={ModelInfo} />
        <Route component={NotFound} />
      </Switch>
    </AppLayout>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="dark">
        <TooltipProvider>
          <Toaster />
          <Router />
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
